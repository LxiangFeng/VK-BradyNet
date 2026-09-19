# test_net.py
import os
import torch
import numpy as np
from slowfast.utils import logging
from slowfast.utils import misc
from slowfast.utils.env import setup_environment
from slowfast.models import build_model
from slowfast.datasets import build_loader
from slowfast.config.defaults import get_cfg
from slowfast.utils.checkpoint import load_checkpoint
import slowfast.utils.distributed as du

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(val_loader, model, cfg):
    """Evaluate one epoch on validation/test set."""
    model.eval()
    running_top1 = 0.0
    running_top5 = 0.0
    num_iters = 0

    # 混淆矩阵
    num_classes = cfg.MODEL.NUM_CLASSES
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
        inputs, labels, index, time, meta = misc._move_to_device(
            inputs, labels, index, time, meta, use_gpu=cfg.NUM_GPUS > 0
        )

        # =========================
        # 前向传播，支持融合模型
        # =========================
        if cfg.DETECTION.ENABLE:
            preds = model(inputs, meta["boxes"])
        elif hasattr(cfg, "FUSION") and cfg.FUSION.ENABLE:
            preds = model(inputs, meta.get("keypoints", None))
        else:
            preds = model(inputs)

        top1_err, top5_err = misc._safe_top_errors(preds, labels)

        # =========================
        # 混淆矩阵
        # =========================
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
        "Test finished: top1_err {:.2f}, top5_err {:.2f}, top1_acc {:.2f}".format(
            avg_top1, avg_top5, 100.0 - avg_top1
        )
    )

    # 打印混淆矩阵
    print("\n" + "=" * 80)
    print("Test Confusion Matrix")
    print("Rows = True Label, Columns = Predicted Label")
    print("      " + " ".join([f"P{i:>6}" for i in range(num_classes)]))
    for i in range(num_classes):
        row = " ".join([f"{conf_mat[i, j]:7d}" for j in range(num_classes)])
        print(f"T{i:<3} {row}")
    print("=" * 80 + "\n")

    return avg_top1, avg_top5

def test(cfg):
    """Full testing pipeline."""
    setup_environment()
    val_loader = build_loader(cfg, "val")

    model = build_model(cfg)
    load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, model, None, use_gpu=cfg.NUM_GPUS>0)

    # =========================
    # 跳过 FLOPs 统计，避免报错
    # =========================
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        model.eval()
        try:
            flops, params = misc.log_model_info(model, cfg, use_train_input=False)
        except Exception as e:
            logger.warning("Skip FLOPs/model stats during test because fusion model needs keypoints: {}".format(e))
            flops, params = 0.0, 0.0

    perform_test(val_loader, model, cfg)

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "MVITv2_S_PD_3cls_rgb_keypoint_fusion.yaml"))
    cfg.NUM_GPUS = 1
    test(cfg)