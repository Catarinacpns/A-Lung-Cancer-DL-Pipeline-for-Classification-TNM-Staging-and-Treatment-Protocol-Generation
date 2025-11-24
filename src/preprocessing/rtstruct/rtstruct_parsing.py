import numpy as np


def extract_tumor_bboxes(rtstruct, roi_map, ct_slices, roi_name="GTV-1"):
    """Extracts tumor contours from RTSTRUCT and converts them into bounding boxes."""
    bboxes = []
    
    # Find the ROINumber for the given ROIName
    roi_number = None
    for num, name in roi_map.items():
        if name == roi_name:
            roi_number = num
            break
    
    if roi_number is None:
        print(f"ROI '{roi_name}' not found!")
        return None
    
    for roi in rtstruct.ROIContourSequence:
        if roi.ReferencedROINumber == roi_number:
            for contour in roi.ContourSequence:
                points = np.array(contour.ContourData).reshape(-1, 3)  # Extract (x, y, z)
                z_pos = points[0, 2]  # Get the Z slice position
                
                # Find closest slice
                z_slice_idx = min(range(len(ct_slices)), key=lambda i: abs(ct_slices[i].ImagePositionPatient[2] - z_pos))
                pixel_spacing = np.array(ct_slices[z_slice_idx].PixelSpacing)
                origin = np.array(ct_slices[z_slice_idx].ImagePositionPatient[:2])
                
                # Convert contour points to pixel coordinates
                points[:, :2] = (points[:, :2] - origin) / pixel_spacing[::-1]  # Swap order for correct mapping
                x_min, y_min = np.min(points[:, :2], axis=0)
                x_max, y_max = np.max(points[:, :2], axis=0)
                
                # Store bounding box
                bboxes.append((z_slice_idx, int(x_min), int(y_min), int(x_max), int(y_max)))
    
    return bboxes