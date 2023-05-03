"""
Usage: python glare_reduction.py
"""
import os
import glob
import math
import cv2
import numpy as np
from PIL import Image, ImageEnhance


# Image conversion
def cv2_to_pil(cv2_im: np.ndarray):
    """
    Convert cv2 BGR image to PIL image
    Argument:
        cv2_im(np.ndarray): cv2 BGR
    Return:
        pil_im(PIL.Image.Image): PIL Image
    """
    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    print(type(cv2_im), type(pil_im))
    return pil_im


def pil_to_cv2(img: object):
    """
    Convert PIL image to cv2 BGR image
    Argument:
        img(PIL.Image.Image): PIL Image
    Return:
        cv2(np.ndarray): BGR Image
    """
    cv_image = np.array(img.convert("RGB"))
    cv_image = cv_image[:, :, ::-1].copy()
    return cv_image


# Polynomial Functions
def first_polynomial_function(image: np.ndarray):
    """
    Implementation of first polynomial function.
    Argument:
        image(np.ndarray): cv2 BGR
    """
    table = np.array(
        [
            1.657766 * i - 0.009157128 * (i**2) + 0.00002579473 * (i**3)
            for i in np.arange(0, 256)
        ]
    ).astype("uint8")
    return cv2.LUT(image, table)


def second_polynomial_function(image: np.ndarray):
    """
    Implementation of second polynomial function.
    Argument:
        image(np.ndarray): cv2 BGR
    """

    table = np.array(
        [
            -4.263256 * math.exp(-14)
            + 1.546429 * i
            - 0.005558036 * (i**2)
            + 0.00001339286 * (i**3)
            for i in np.arange(0, 256)
        ]
    ).astype("uint8")
    return cv2.LUT(image, table)


# Gamma Correction
def adjust_gamma(image: np.ndarray, gamma=1.0):
    """
    Implementation of gamma correction
    Argument:
        image(np.ndarray): cv2 BGR
    """

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv2.LUT(image, table)


# Contrast & Brightness
def enhance_contrast(image: np.ndarray, factor=1.4):
    """
    Enhance contrast base on PIL->ImageEnhance.
    Default suitable factor is 1,4.
    Argument:
        image(np.ndarray): cv2 BGR
        factor: contrast's factor

    Return:
        (np.ndarray): Enhanced contrast BGR image
    """

    _image = ImageEnhance.Contrast(cv2_to_pil(image)).enhance(factor)

    return pil_to_cv2(_image)


# Methods
def reduce_glare(image: np.ndarray):
    """
    Mixed 4 filter:
        1. First polynomial function
        2. Gamma correction: g = 0.75
        3. Second polynomial function
        4. Gamma correction: g = 0.8
    Argument:
        image(np.ndarray): cv2 BGR
    Return:
        (np.ndarray): Reduced Glare
    """
    _image = adjust_gamma(
        second_polynomial_function(
            adjust_gamma(first_polynomial_function(image), 0.75)
        ),
        0.8,
    )
    return _image


def mix_filter(image: np.ndarray):
    """
    Mixed 4 steps:
        1. Reduce glare
        2. Enhance contract: f = 1.6
        3. Reduce glare
        4. Enhance contract: f = 1.4

    Argument:
        image(np.ndarray): cv2 BGR
    Return:
        (np.ndarray): Reduced glare & Clearer image
    """
    _image = enhance_contrast(
        reduce_glare(enhance_contrast(reduce_glare(image), factor=1.6)), factor=1.4
    )
    return _image


if __name__ == "__main__":
    EXT = "JPG"
    img_files = glob.glob(os.path.join("light-effects", f"*.{EXT}"))
    for process in ["001", "003"]:
        for file_name in img_files:
            basename, _ = os.path.splitext(os.path.basename(file_name))
            image = cv2.imread(file_name)
            if process == "001":
                _image = reduce_glare(image)
            elif process == "003":
                _image = mix_filter(image)
            cv2.imwrite(
                os.path.join("light-effects-output", f"{basename}_{process}.{EXT}"),
                _image,
            )
