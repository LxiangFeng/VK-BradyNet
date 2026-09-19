#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random

import numpy as np
import pandas
import slowfast.utils.logging as logging
import torch
import torch.utils.data
from slowfast.utils.env import pathmgr
from torchvision import transforms

from . import (
    decoder as decoder,
    transform as transform,
    utils as utils,
    video_container as container,
)
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment, MaskingGenerator, MaskingGenerator3D

logger = logging.get_logger(__name__)


def _as_int(value, default=1):
    """
    Convert config values that may accidentally be list/tuple/None into int.

    This protects decoder.decode(..., num_clips=..., num_frames=..., sampling_rate=...)
    from receiving [] or [x], which causes errors such as:
        unsupported operand type(s) for -: 'list' and 'int'
    """
    if value is None:
        return int(default)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return int(default)
        return int(value[0])
    return int(value)


def _to_time_idx_array(time_idx_item):
    """Normalize decoder time_idx metadata to a numpy float32 array."""
    if time_idx_item is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if torch.is_tensor(time_idx_item):
        return time_idx_item.detach().cpu().numpy().astype(np.float32)
    return np.asarray(time_idx_item, dtype=np.float32)


class _ScalarList(list):
    """
    A one-element list that also behaves like a scalar number.

    Some SlowFast decoder.py versions do both of the following:
        assert len(sampling_rate) == len(num_frames)
        clip_size = sampling_rate * num_frames - 1

    A normal int fails len(...), while a normal list fails arithmetic.
    This wrapper keeps compatibility with both code paths.
    """
    def __init__(self, value):
        super().__init__([int(value)])

    @property
    def value(self):
        return int(self[0])

    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def __index__(self):
        return self.value

    def _other(self, other):
        if isinstance(other, _ScalarList):
            return other.value
        if isinstance(other, (list, tuple)):
            if len(other) == 0:
                return 0
            return other[0]
        return other

    def __add__(self, other):
        return self.value + self._other(other)

    def __radd__(self, other):
        return self._other(other) + self.value

    def __sub__(self, other):
        return self.value - self._other(other)

    def __rsub__(self, other):
        return self._other(other) - self.value

    def __mul__(self, other):
        return self.value * self._other(other)

    def __rmul__(self, other):
        return self._other(other) * self.value

    def __truediv__(self, other):
        return self.value / self._other(other)

    def __rtruediv__(self, other):
        return self._other(other) / self.value

    def __floordiv__(self, other):
        return self.value // self._other(other)

    def __rfloordiv__(self, other):
        return self._other(other) // self.value

    def __lt__(self, other):
        return self.value < self._other(other)

    def __le__(self, other):
        return self.value <= self._other(other)

    def __gt__(self, other):
        return self.value > self._other(other)

    def __ge__(self, other):
        return self.value >= self._other(other)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
        )
        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = max(
                1,
                _as_int(cfg.TEST.NUM_ENSEMBLE_VIEWS, 1)
                * _as_int(cfg.TEST.NUM_SPATIAL_CROPS, 1),
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._path_to_keypoints = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS

        with pathmgr.open(path_to_file, "r") as f:
            if self.use_chunk_loading:
                rows = self._get_chunk(f, self.cfg.DATA.LOADER_CHUNK_SIZE)
            else:
                rows = f.read().splitlines()
            for clip_idx, path_label in enumerate(rows):
                fetch_info = path_label.strip().split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                fetch_info = [x for x in fetch_info if x != ""]
                # Supported CSV formats:
                #   video_path label
                #   video_path keypoint_path label   (when FUSION.ENABLE=True)
                if len(fetch_info) == 2:
                    path, label = fetch_info
                    kpt_path = ""
                elif len(fetch_info) == 3:
                    if hasattr(self.cfg, "FUSION") and self.cfg.FUSION.ENABLE:
                        path, kpt_path, label = fetch_info
                    else:
                        path, _unused, label = fetch_info
                        kpt_path = ""
                elif len(fetch_info) == 1:
                    path, label = fetch_info[0], 0
                    kpt_path = ""
                else:
                    raise RuntimeError(
                        "Failed to parse video fetch {} info {} retries.".format(
                            path_to_file, fetch_info
                        )
                    )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    if hasattr(self.cfg, "FUSION") and self.cfg.FUSION.ENABLE:
                        if kpt_path:
                            if os.path.isabs(kpt_path):
                                full_kpt_path = kpt_path
                            elif self.cfg.FUSION.KEYPOINT_ROOT:
                                full_kpt_path = os.path.join(self.cfg.FUSION.KEYPOINT_ROOT, kpt_path)
                            else:
                                full_kpt_path = os.path.join(self.cfg.DATA.PATH_PREFIX, kpt_path)
                        else:
                            full_kpt_path = ""
                        self._path_to_keypoints.append(full_kpt_path)
                    else:
                        self._path_to_keypoints.append("")
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert len(self._path_to_videos) > 0, (
            "Failed to load Kinetics split {} from {}".format(
                self._split_idx, path_to_file
            )
        )
        logger.info(
            "Constructing kinetics dataloader (size: {} skip_rows {}) from {} ".format(
                len(self._path_to_videos), self.skip_rows, path_to_file
            )
        )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _get_chunk(self, path_to_file, chunksize):
        try:
            chunk = next(
                pandas.read_csv(
                    path_to_file,
                    chunksize=self.cfg.DATA.LOADER_CHUNK_SIZE,
                    skiprows=self.skip_rows,
                )
            )
        except Exception:
            self.skip_rows = 0
            return self._get_chunk(path_to_file, chunksize)
        else:
            return pandas.array(chunk.values.flatten(), dtype="string")


    def _sample_keypoints_to_fixed_length(self, arr):
        """Convert keypoint array to shape [T, K*C] with fixed T."""
        target_t = int(self.cfg.FUSION.KEYPOINT_NUM_FRAMES)
        num_k = int(self.cfg.FUSION.NUM_KEYPOINTS)
        dim = int(self.cfg.FUSION.KEYPOINT_DIM)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            # [T, K*C] -> [T, K, C]
            if arr.shape[1] >= num_k * dim:
                arr = arr[:, :num_k * dim].reshape(arr.shape[0], num_k, dim)
            else:
                padded = np.zeros((arr.shape[0], num_k * dim), dtype=np.float32)
                padded[:, :arr.shape[1]] = arr
                arr = padded.reshape(arr.shape[0], num_k, dim)
        elif arr.ndim == 3:
            # [T, K, C]
            if arr.shape[1] < num_k:
                padded = np.zeros((arr.shape[0], num_k, max(arr.shape[2], dim)), dtype=np.float32)
                padded[:, :arr.shape[1], :arr.shape[2]] = arr
                arr = padded
            arr = arr[:, :num_k, :dim]
        else:
            arr = np.zeros((1, num_k, dim), dtype=np.float32)

        if arr.shape[-1] < dim:
            padded = np.zeros((arr.shape[0], num_k, dim), dtype=np.float32)
            padded[:, :, :arr.shape[-1]] = arr
            arr = padded

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        t = arr.shape[0]
        if t <= 0:
            arr = np.zeros((target_t, num_k, dim), dtype=np.float32)
        elif t == target_t:
            pass
        elif t > target_t:
            ids = np.linspace(0, t - 1, target_t).astype(np.int64)
            arr = arr[ids]
        else:
            pad = np.repeat(arr[-1:], target_t - t, axis=0)
            arr = np.concatenate([arr, pad], axis=0)

        return arr.reshape(target_t, num_k * dim).astype(np.float32)

    def _load_keypoints(self, index):
        """Load keypoint npy for one video. Returns tensor [T, K*C]."""
        target_t = int(self.cfg.FUSION.KEYPOINT_NUM_FRAMES)
        feat_dim = int(self.cfg.FUSION.NUM_KEYPOINTS) * int(self.cfg.FUSION.KEYPOINT_DIM)
        zero = np.zeros((target_t, feat_dim), dtype=np.float32)

        if not (hasattr(self.cfg, "FUSION") and self.cfg.FUSION.ENABLE):
            return torch.from_numpy(zero)

        kpt_path = self._path_to_keypoints[index] if index < len(self._path_to_keypoints) else ""
        if not kpt_path or not os.path.exists(kpt_path):
            if self.cfg.FUSION.MISSING_KEYPOINT == "error":
                raise FileNotFoundError("Keypoint npy not found: {}".format(kpt_path))
            return torch.from_numpy(zero)

        try:
            arr = np.load(kpt_path)
            arr = self._sample_keypoints_to_fixed_length(arr)
            return torch.from_numpy(arr)
        except Exception as e:
            if self.cfg.FUSION.MISSING_KEYPOINT == "error":
                raise RuntimeError("Failed to load keypoints {}: {}".format(kpt_path, e))
            return torch.from_numpy(zero)

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index
        if self.dummy_output is not None:
            return self.dummy_output
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))
        num_decode = _as_int(
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode in ["train"] else 1,
            default=1,
        )
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                num_decode - len(min_scale)
            )
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE or self.cfg.MULTIGRID.SHORT_CYCLE
                else [self.cfg.DATA.TRAIN_CROP_SIZE] * (num_decode - len(crop_size))
            )
            assert self.mode in ["train", "val"]
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
                if self.mode not in ["test"]:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue  # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )

            base_sampling_rate = utils.get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )

            if self.mode in ["train"]:
                assert len(min_scale) == len(max_scale) == len(crop_size) == num_decode

            target_fps = self.cfg.DATA.TARGET_FPS
            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(0.0, self.cfg.DATA.TRAIN_JITTER_FPS)

            # Decode video.
            #
            # Important:
            #   Some SlowFast decoder versions expect scalar int values for
            #   sampling_rate, num_frames, and num_clips. Passing [x] or []
            #   can trigger:
            #       unsupported operand type(s) for -: 'list' and 'int'
            #
            # Therefore this fusion loader always passes scalar values into
            # decoder.decode and stores the decoded clip(s) back into lists.
            for decode_idx in range(num_decode):
                cur_num_frames = _as_int(self.cfg.DATA.NUM_FRAMES, default=16)
                cur_sampling_rate = _as_int(base_sampling_rate, default=1)
                cur_num_clips = (
                    _as_int(self.cfg.TEST.NUM_ENSEMBLE_VIEWS, default=1)
                    if self.mode in ["test"]
                    else 1
                )
                cur_temporal_sample_index = _as_int(
                    temporal_sample_index,
                    default=-1 if self.mode in ["train", "val"] else 0,
                )

                try:
                    frames, time_idx, _ = decoder.decode(
                        video_container,
                        _ScalarList(cur_sampling_rate),
                        _ScalarList(cur_num_frames),
                        cur_temporal_sample_index,
                        cur_num_clips,
                        video_meta=(
                            self._video_meta[index] if len(self._video_meta) < 5e6 else {}
                        ),  # do not cache on huge datasets
                        target_fps=target_fps,
                        backend=self.cfg.DATA.DECODING_BACKEND,
                        use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                        max_spatial_scale=min_scale[decode_idx],
                        time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
                        temporally_rnd_clips=True,
                        min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                        max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
                    )
                except TypeError as e:
                    # Some older SlowFast decoder implementations do not expose
                    # the extended keyword arguments used by newer versions.
                    # Fall back to the common minimal signature.
                    if "unexpected keyword" not in str(e):
                        raise
                    frames, time_idx, _ = decoder.decode(
                        video_container,
                        _ScalarList(cur_sampling_rate),
                        _ScalarList(cur_num_frames),
                        cur_temporal_sample_index,
                        cur_num_clips,
                        video_meta=(
                            self._video_meta[index] if len(self._video_meta) < 5e6 else {}
                        ),
                        target_fps=target_fps,
                        backend=self.cfg.DATA.DECODING_BACKEND,
                        use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                        max_spatial_scale=min_scale[decode_idx],
                    )

                # Normalize decoder outputs.
                # Newer code paths may return a list; older ones may return
                # a single tensor. Downstream code expects one tensor per
                # decode_idx.
                if isinstance(frames, (list, tuple)):
                    frames_decoded[decode_idx] = frames[0] if len(frames) > 0 else None
                else:
                    frames_decoded[decode_idx] = frames

                if isinstance(time_idx, (list, tuple)):
                    time_idx_decoded[decode_idx] = (
                        time_idx[0] if len(time_idx) > 0 else None
                    )
                elif torch.is_tensor(time_idx) and time_idx.ndim >= 2:
                    time_idx_decoded[decode_idx] = time_idx[0]
                else:
                    time_idx_decoded[decode_idx] = time_idx

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames_decoded is None or any(x is None for x in frames_decoded):
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if (
                    self.mode not in ["test"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
                if self.mode in ["train"]
                else 1
            )
            num_out = num_aug * num_decode
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]

            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    f_out[idx] = frames_decoded[i].clone()

                    # time_idx_decoded is auxiliary metadata for classification.
                    # Keep numpy arrays here; do not store torch.Tensor.
                    if time_idx_decoded is not None and i < len(time_idx_decoded):
                        time_idx_out[idx] = _to_time_idx_array(time_idx_decoded[i])
                    else:
                        time_idx_out[idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                    f_out[idx] = f_out[idx].float()
                    f_out[idx] = f_out[idx] / 255.0

                    if self.mode in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
                        f_out[idx] = transform.color_jitter_video_ssl(
                            f_out[idx],
                            bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                            hue=self.cfg.DATA.SSL_COLOR_HUE,
                            p_convert_gray=self.p_convert_gray,
                            moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                            gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                            gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                        )

                    if self.aug and self.cfg.AUG.AA_TYPE:
                        aug_transform = create_random_augment(
                            input_size=(f_out[idx].size(1), f_out[idx].size(2)),
                            auto_augment=self.cfg.AUG.AA_TYPE,
                            interpolation=self.cfg.AUG.INTERPOLATION,
                        )
                        # T H W C -> T C H W.
                        f_out[idx] = f_out[idx].permute(0, 3, 1, 2)
                        list_img = self._frame_to_list_img(f_out[idx])
                        list_img = aug_transform(list_img)
                        f_out[idx] = self._list_img_to_frames(list_img)
                        f_out[idx] = f_out[idx].permute(0, 2, 3, 1)

                    # Perform color normalization.
                    f_out[idx] = utils.tensor_normalize(
                        f_out[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )

                    # T H W C -> C T H W.
                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                    scl, asp = (
                        self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                        self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                    )
                    relative_scales = (
                        None if (self.mode not in ["train"] or len(scl) == 0) else scl
                    )
                    relative_aspect = (
                        None if (self.mode not in ["train"] or len(asp) == 0) else asp
                    )
                    f_out[idx] = utils.spatial_sampling(
                        f_out[idx],
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale[i],
                        max_scale=max_scale[i],
                        crop_size=crop_size[i],
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                        aspect_ratio=relative_aspect,
                        scale=relative_scales,
                        motion_shift=(
                            self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                            if self.mode in ["train"]
                            else False
                        ),
                    )

                    if self.rand_erase:
                        erase_transform = RandomErasing(
                            self.cfg.AUG.RE_PROB,
                            mode=self.cfg.AUG.RE_MODE,
                            max_count=self.cfg.AUG.RE_COUNT,
                            num_splits=self.cfg.AUG.RE_COUNT,
                            device="cpu",
                        )
                        f_out[idx] = erase_transform(
                            f_out[idx].permute(1, 0, 2, 3)
                        ).permute(1, 0, 2, 3)

                    f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
                    if self.cfg.AUG.GEN_MASK_LOADER:
                        mask = self._gen_mask()
                        f_out[idx] = f_out[idx] + [torch.Tensor(), mask]
            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.asarray(time_idx_out, dtype=np.float32)
            if (
                num_aug * num_decode > 1
                and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode
            meta = {}
            sample_index = index[0] if isinstance(index, list) else index
            if hasattr(self.cfg, "FUSION") and self.cfg.FUSION.ENABLE:

                meta["keypoints"] = self._load_keypoints(sample_index)

                if not hasattr(self, "_debug_printed"):
                    print("=" * 60)
                    print("DEBUG KEYPOINT:")
                    print(
                        "path:",
                        self._path_to_keypoints[sample_index]
                    )
                    print(
                        "shape:",
                        meta["keypoints"].shape
                    )
                    print(
                        "mean:",
                        meta["keypoints"].mean()
                    )
                    print(
                        "std:",
                        meta["keypoints"].std()
                    )
                    print(
                        "abs_sum:",
                        meta["keypoints"].abs().sum()
                    )
                    print("=" * 60)

                    self._debug_printed = True

                meta["keypoint_path"] = (
                    self._path_to_keypoints[sample_index]
                    if sample_index < len(self._path_to_keypoints)
                    else ""
                )
            if self.cfg.DATA.DUMMY_LOAD:
                if self.dummy_output is None:
                    self.dummy_output = (frames, label, index, time_idx, meta)
            return frames, label, index, time_idx, meta
        else:
            last_path = (
                self._path_to_videos[index]
                if isinstance(index, int) and index < len(self._path_to_videos)
                else "unknown"
            )
            raise RuntimeError(
                "Failed to fetch/decode video after {} retries. Last index: {}, path: {}".format(
                    self._num_retries, index, last_path
                )
            )

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=int)
            n_mask = round(self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO)
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
