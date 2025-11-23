# Imports
import pydicom
import cv2
import numpy as np

def read_dicom_image(image_path):
    dicom = pydicom.dcmread(image_path)
    img = dicom.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return img
