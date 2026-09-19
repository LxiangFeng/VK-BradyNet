#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train a video classification model."""

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

# 本地导入，不要用 vision.fair.slowfast
# 不要在顶部导入 test_net，否则即使不测试也会触发 test_net 的依赖错误
from tools.train_net import train


def main():
    """
    Main function to spawn the train process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))

    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        # 执行训练
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

        # 不在本脚本中执行测试 / 验证
        # 你已经写了单独的验证脚本，所以这里直接跳过
        if cfg.TEST.ENABLE:
            print(
                "[INFO] cfg.TEST.ENABLE=True, but built-in test is disabled in this training wrapper. "
                "Please use your separate validation script."
            )


if __name__ == "__main__":
    main()