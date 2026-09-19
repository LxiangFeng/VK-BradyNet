#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simplified and fixed train_net.py for SlowFast video classification.

This version is intended for a normal video classification task such as:
- MODEL.NUM_CLASSES = 3
- DATASET = kinetics
- DETECTION.ENABLE = False
- MASK.ENABLE = False
- DEMO.ENABLE = False

Fixes:
1. Does not calculate invalid top-5 when NUM_CLASSES < 5.
2. Keeps the public entry function train(cfg), so tools/run_net.py can import it.
3. Supports CUDA, AMP, optimizer, train loader, val loader, and checkpoints.
"""

import math
import pprint
import numpy as np
import torch

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.metrics as metrics

from slowfast.datasets import loader
from slowfast.models import build_model

logger = logging.get_logger(__name__)


def _move_to_device(inputs, labels, index, time, meta, use_gpu=True):
    """Move a SlowFast batch to GPU if needed."""
    if not use_gpu:
        return inputs, labels, index, time, meta

    if isinstance(inputs, list):
        for i in range(len(inputs)):
            if isinstance(inputs[i], list):
                for j in range(len(inputs[i])):
                    inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
            else:
                inputs[i] = inputs[i].cuda(non_blocking=True)
    else:
        inputs = inputs.cuda(non_blocking=True)

    if torch.is_tensor(labels):
        labels = labels.cuda(non_blocking=True)

    if torch.is_tensor(index):
        index = index.cuda(non_blocking=True)

    if torch.is_tensor(time):
        time = time.cuda(non_blocking=True)

    if isinstance(meta, dict):
        for key, val in meta.items():
            if torch.is_tensor(val):
                meta[key] = val.cuda(non_blocking=True)
            elif isinstance(val, list):
                for i in range(len(val)):
                    if torch.is_tensor(val[i]):
                        val[i] = val[i].cuda(non_blocking=True)

    return inputs, labels, index, time, meta


def _batch_size(inputs):
    """Get batch size from SlowFast style inputs."""
    if isinstance(inputs, list):
        if len(inputs) > 0 and isinstance(inputs[0], list):
            return inputs[0][0].size(0)
        return inputs[0].size(0)
    return inputs.size(0)


def _safe_top_errors(preds, labels):
    """
    Return top1_err and top5_err safely.

    For small NUM_CLASSES, top5 is invalid. Set top5_err = top1_err
    """
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)

    num_classes = preds.size(1)
    ks = (1, 5) if num_classes >= 5 else (1,)
    topk_errs = metrics.topk_errors(preds, labels, ks)

    top1_err = topk_errs[0]
    top5_err = topk_errs[1] if len(topk_errs) > 1 else top1_err
    return top1_err, top5_err


def train_epoch(train_loader, model, optimizer, scaler, cur_epoch, cfg):
    """Train one epoch."""
    model.train()
    data_size = len(train_loader)
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    num_iters = 0

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(train_loader):
        inputs, labels, index, time, meta = _move_to_device(
            inputs, labels, index, time, meta, use_gpu=cfg.NUM_GPUS > 0
        )

        epoch_exact = cur_epoch + float(cur_iter) / max(data_size, 1)
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            if cfg.DETECTION.ENABLE:
                preds = model(inputs, meta["boxes"])
            elif hasattr(cfg, "FUSION") and cfg.FUSION.ENABLE:
                preds = model(inputs, meta.get("keypoints", None))
            else:
                preds = model(inputs)

            loss = loss_fun(preds, labels)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]

        misc.check_nan_losses(loss)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL)
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            top1_err, top5_err = _safe_top_errors(preds.detach(), labels.detach())

        running_loss += float(loss.detach().item())
        running_top1 += float(top1_err.detach().item() if torch.is_tensor(top1_err) else top1_err)
        running_top5 += float(top5_err.detach().item() if torch.is_tensor(top5_err) else top5_err)
        num_iters += 1

        if cur_iter % max(int(cfg.LOG_PERIOD), 1) == 0:
            logger.info(
                "epoch {:03d} iter {:05d}/{:05d}, loss {:.4f}, top1_err {:.2f}, top5_err {:.2f}, lr {:.6g}".format(
                    cur_epoch + 1,
                    cur_iter,
                    data_size,
                    running_loss / max(num_iters, 1),
                    running_top1 / max(num_iters, 1),
                    running_top5 / max(num_iters, 1),
                    lr,
                )
            )

        if cfg.NUM_GPUS > 0:
            torch.cuda.empty_cache()

    avg_loss = running_loss / max(num_iters, 1)
    avg_top1 = running_top1 / max(num_iters, 1)
    avg_top5 = running_top5 / max(num_iters, 1)
    logger.info(
        "Train epoch {} finished: loss {:.4f}, top1_err {:.2f}, top5_err {:.2f}".format(
            cur_epoch + 1, avg_loss, avg_top1, avg_top5
        )
    )
    return avg_loss, avg_top1, avg_top5


@torch.no_grad()
def eval_epoch(val_loader, model, cur_epoch, cfg):
    """Evaluate one epoch and print confusion matrix in console."""
    model.eval()

    running_top1 = 0.0
    running_top5 = 0.0
    num_iters = 0

    # 初始化混淆矩阵
    num_classes = cfg.MODEL.NUM_CLASSES
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
        inputs, labels, index, time, meta = _move_to_device(
            inputs, labels, index, time, meta, use_gpu=cfg.NUM_GPUS > 0
        )

        if cfg.DETECTION.ENABLE:
            preds = model(inputs, meta["boxes"])
        elif hasattr(cfg, "FUSION") and cfg.FUSION.ENABLE:
            preds = model(inputs, meta.get("keypoints", None))
        else:
            preds = model(inputs)

        top1_err, top5_err = _safe_top_errors(preds, labels)

        # 累计混淆矩阵
        if labels.ndim > 1:
            true_cls = labels.argmax(dim=1)
        else:
            true_cls = labels

        pred_cls = preds.argmax(dim=1)

        true_cls = true_cls.detach().cpu().numpy()
        pred_cls = pred_cls.detach().cpu().numpy()

        for t, p in zip(true_cls, pred_cls):
            t = int(t)
            p = int(p)
            if 0 <= t < num_classes and 0 <= p < num_classes:
                conf_mat[t, p] += 1

        running_top1 += float(top1_err.detach().item() if torch.is_tensor(top1_err) else top1_err)
        running_top5 += float(top5_err.detach().item() if torch.is_tensor(top5_err) else top5_err)
        num_iters += 1

    avg_top1 = running_top1 / max(num_iters, 1)
    avg_top5 = running_top5 / max(num_iters, 1)

    logger.info(
        "Val epoch {} finished: top1_err {:.2f}, top5_err {:.2f}, top1_acc {:.2f}".format(
            cur_epoch + 1, avg_top1, avg_top5, 100.0 - avg_top1
        )
    )

    # 打印混淆矩阵
    print("\n" + "=" * 80)
    print("Validation Confusion Matrix")
    print("Rows = True Label, Columns = Predicted Label")
    print("      " + " ".join([f"P{i:>6}" for i in range(num_classes)]))
    for i in range(num_classes):
        row = " ".join([f"{conf_mat[i, j]:7d}" for j in range(num_classes)])
        print(f"T{i:<3} {row}")
    print("=" * 80 + "\n")

    return avg_top1, avg_top5


def train(cfg):
    """
    Entry point used by tools/run_net.py.
    """
    du.init_distributed_training(cfg)

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)

    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    if cfg.DETECTION.ENABLE:
        raise NotImplementedError(
            "This simplified train_net.py is for classification. Please set DETECTION.ENABLE: False."
        )

    if cfg.MASK.ENABLE:
        raise NotImplementedError(
            "This simplified train_net.py is for classification. Please set MASK.ENABLE: False."
        )

    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        try:
            misc.log_model_info(model, cfg, use_train_input=True)
        except Exception as e:
            logger.warning("log_model_info failed but training will continue: {}".format(e))

    optimizer = optim.construct_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    start_epoch = 0

    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1

    train_loader = loader.construct_loader(cfg, "train")
    # Validation during training should not depend on TEST.ENABLE.
    # TEST.ENABLE controls the standalone test stage in run_net.py, while
    # TRAIN.EVAL_PERIOD controls validation frequency during training.
    val_loader = loader.construct_loader(cfg, "val")

    logger.info("Start epoch: {}".format(start_epoch + 1))

    best_top1 = float("inf")

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        loader.shuffle_dataset(train_loader, cur_epoch)

        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)

        train_loss, train_top1, train_top5 = train_epoch(
            train_loader, model, optimizer, scaler, cur_epoch, cfg
        )

        is_checkp_epoch = (
            (cur_epoch + 1) % int(cfg.TRAIN.CHECKPOINT_PERIOD) == 0
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )

        is_eval_epoch = (
            (cur_epoch + 1) % int(cfg.TRAIN.EVAL_PERIOD) == 0
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )

        if is_eval_epoch and val_loader is not None:
            top1_err, top5_err = eval_epoch(val_loader, model, cur_epoch, cfg)
            if top1_err < best_top1:
                best_top1 = top1_err
                logger.info("New best top1_acc: {:.2f}".format(100.0 - best_top1))

        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

    logger.info("Training done. Best top1_acc: {:.2f}".format(100.0 - best_top1))
    return "training done"