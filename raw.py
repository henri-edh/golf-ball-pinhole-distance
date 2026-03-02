import math
from pathlib import Path

import cv2
import numpy as np


def _unpacked10(
        packed: np.ndarray,
        h: int, w: int,
        hdr: bool = False,
        gain: int = 1
) -> np.ndarray:
    assert packed.dtype == np.uint8
    byte_count = np.prod(packed.shape)
    assert byte_count % 5 == 0
    assert byte_count == h * w * 5 // 4

    # New (apply gain before 10->8 bit)
    bytes_ = packed.astype(np.uint16).reshape(-1, 5)
    upper = bytes_[:, :4]
    lower = bytes_[:, 4].reshape(-1, 1)
    shorts = (upper << 2) + (lower >> np.array([6, 4, 2, 0], dtype=np.uint16).reshape((1, 4)) & 3)
    if hdr:
        return (gain * shorts).reshape(h, w)
    else:
        # Note: cleverly adjusts gain before reducing to 8 bit
        return np.minimum(gain * shorts // 4, 255).astype(np.uint8).reshape(h, w)


def load_raw(
        filepath: Path | str,
        hdr: bool = False,
        grayscale: bool = False,
        float32: bool = False,
        gain: int = 1,
        expected_shape: tuple[int, int] | tuple[int, int, int] | None = None
) -> np.ndarray:
    packed = np.fromfile(filepath, dtype=np.uint8)
    if expected_shape is None:
        if len(packed) == 1280 * 720 * 5 // 4:
            h, w = 720, 1280
        elif len(packed) == 640 * 480 * 5 // 4:
            h, w = 480, 640
        else:
            print(filepath)
            raise NotImplementedError
    else:
        h, w, *_ = expected_shape
        if len(packed) != h * w * 5 // 4:
            # Return zeros array for corrupted images (for possible viewing)
            dtype = np.float32 if float32 else (np.uint16 if hdr else np.uint8)
            return np.zeros((h, w, 3), dtype=dtype)
    if grayscale:
        img = cv2.cvtColor(
            _unpacked10(packed, h, w, hdr=hdr, gain=gain),
            cv2.COLOR_BAYER_RG2GRAY
        )
    else:
        img = cv2.cvtColor(
            _unpacked10(packed, h, w, hdr=hdr, gain=gain),
            cv2.COLOR_BAYER_RG2RGB
        )
    if float32:
        return img / 1023 if hdr else img / 255
    else:
        return img


def gvp_probe_raw(
        filepath: Path | str,
        expected_shape: tuple[int, int] | tuple[int, int, int] | None = None
) -> np.ndarray:  # returns LUT
    packed = np.fromfile(filepath, dtype=np.uint8)
    if expected_shape is None:
        if len(packed) == 1280 * 720 * 5 // 4:
            h, w = 720, 1280
        elif len(packed) == 640 * 480 * 5 // 4:
            h, w = 480, 640
        else:
            raise NotImplementedError
    else:
        h, w, *_ = expected_shape
        if len(packed) != h * w * 5 // 4:
            # raise error for corrupted images (for critical testing or dataset generation)
            raise NotImplementedError
    raw10 = _unpacked10(packed, h, w, hdr=True, gain=1)
    # print(f'{raw10.shape=}, {raw10.dtype=}')

    lut = np.zeros((1 << 16,), dtype=np.uint8)

    # region GVP Equivalent LUT

    # See RawConverter::probeRawImageWithShape() in rawconverter.cpp in GolfVideoProcessor for
    # the source of the following logic

    # Arguments: [img], [channel_index], mask, [bins], [range]
    full_histogram = (cv2.calcHist([raw10], [0], None, [1024], [0, 1024])
                      .flatten().astype(np.uint16))
    # print(f'{full_histogram.shape=}, {full_histogram.dtype=}')
    # print(f'{np.sum(full_histogram)=}')

    full_pixel_count_thresh = math.floor(0.99 * w * h - np.sum(full_histogram[-4:]))
    full_pixel_count = 0
    full_min_idx, full_max_idx = 0, 1024
    for i in range(1024):
        if full_min_idx == 0 and full_histogram[i] > 0:
            full_min_idx = i
        if full_max_idx == 1024 and full_pixel_count > full_pixel_count_thresh:
            full_max_idx = i
        full_pixel_count += int(full_histogram[i])  # note the need for int(), silly numpy
    # print(f'{full_pixel_count_thresh=}')
    # print(f'{full_pixel_count=}')

    full_range = full_max_idx - full_min_idx
    if full_range <= 255:
        for i in range(1024):
            if i < full_min_idx:
                lut[i] = 0
            elif full_min_idx <= i < full_min_idx + 256:
                lut[i] = i - full_min_idx
            else:
                lut[i] = 255
    else:
        for i in range(1024):
            if i < full_min_idx:
                lut[i] = 0
            elif full_min_idx <= i < full_max_idx:
                lut[i] = math.floor(255 * (i - full_min_idx) / full_range + 0.5)
            else:
                lut[i] = 255

    # endregion

    # Complete 16-bit LUT (even though > 1023 will never happen)
    for i in range(1024, len(lut)):
        lut[i] = 255

    return lut


def gvp_load_raw(
        filepath: Path | str,
        lut: np.ndarray,
        expected_shape: tuple[int, int] | tuple[int, int, int] | None = None
) -> np.ndarray:  # returns 8-bit image

    assert lut.ndim == 1
    assert len(lut) == 1 << 16
    assert lut.dtype == np.uint8

    packed = np.fromfile(filepath, dtype=np.uint8)
    if expected_shape is None:
        if len(packed) == 1280 * 720 * 5 // 4:
            h, w = 720, 1280
        elif len(packed) == 640 * 480 * 5 // 4:
            h, w = 480, 640
        else:
            raise NotImplementedError
    else:
        h, w, *_ = expected_shape
        if len(packed) != h * w * 5 // 4:
            # raise error for corrupted images (for critical testing or dataset generation)
            raise NotImplementedError

    return cv2.LUT(_unpacked10(packed, h, w, hdr=True, gain=1), lut)
