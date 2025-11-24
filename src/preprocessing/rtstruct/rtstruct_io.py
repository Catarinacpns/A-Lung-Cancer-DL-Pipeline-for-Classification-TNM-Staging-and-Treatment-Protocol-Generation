import os
import pydicom
import numpy as np
import cv2

def load_rtstruct(rtstruct_path):
    """Loads RTSTRUCT DICOM file and extracts tumor contours"""
    rtstruct = pydicom.dcmread(rtstruct_path)
    
    # Ensure the file is actually an RTSTRUCT file
    if rtstruct.SOPClassUID != "1.2.840.10008.5.1.4.1.1.481.3":
        raise ValueError("File is not an RTSTRUCT DICOM file.")
    
    # Map ROINumber to ROIName
    roi_map = {
        roi.ROINumber: roi.ROIName for roi in rtstruct.StructureSetROISequence
    }
    print("Available ROIs:", roi_map)
    
    return rtstruct, roi_map

def load_ct_images(ct_dir):
    """Loads all CT DICOM images from the given directory and sorts them by InstanceNumber"""
    ct_slices = []
    for file in os.listdir(ct_dir):
        if file.endswith(".dcm"):
            dicom_path = os.path.join(ct_dir, file)
            dicom_data = pydicom.dcmread(dicom_path)
            ct_slices.append(dicom_data)
    
    # Sort slices based on Instance Number to ensure correct order
    ct_slices.sort(key=lambda x: int(x.InstanceNumber))
    print(f"Loaded {len(ct_slices)} CT slices.")
    return ct_slices

def read_dicom_image(dicom):
    """Reads and normalizes a DICOM image while maintaining original contrast"""
    img = dicom.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return img
