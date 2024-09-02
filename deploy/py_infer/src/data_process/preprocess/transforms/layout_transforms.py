import cv2
import numpy as np
from mindocr.data import layout_dataset


def letterbox(scaleup):
    def func(data):
        image = data["image"]
        hw_ori = data["raw_img_shape"]
        new_shape = data["target_size"]
        color = (114, 114, 114)

        data["image"], data["image_ids"], data["hw_ori"], data["hw_scale"], data["pad"] = layout_dataset.letterbox(
            image, 0, hw_ori, new_shape, scaleup, color
        )
        return data

    return func


def image_norm(scale=255.0):
    def func(data):
        image = data["image"]
        data["image"], _ = layout_dataset.image_norm(image, 0, scale)
        return data

    return func


def image_transpose(bgr2rgb=True, hwc2chw=True):
    def func(data):
        image = data["image"]
        data["image"] = layout_dataset.image_transpose(image, 0, bgr2rgb, hwc2chw)
        return data

    return func
