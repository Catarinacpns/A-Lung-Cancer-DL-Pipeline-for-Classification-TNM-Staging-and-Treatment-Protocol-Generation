import os
from src.utils.file_utils import loadFileInformation


def getUID_path(path):
    uid_dict = {}
    for patient_folder in sorted(os.listdir(path)):
        patient_path = os.path.join(path, patient_folder)
        if os.path.isdir(patient_path):
            for root, dirs, files in sorted(os.walk(patient_path)):
                for file in sorted(files):
                    if file.endswith('.dcm'):
                        dicom_path = os.path.join(root, file)
                        try:
                            info = loadFileInformation(dicom_path)
                            uid = info['dicom_num']
                            patient_id = patient_folder.replace('Lung_Dx-', '')
                            description = os.path.basename(os.path.dirname(os.path.dirname(dicom_path)))
                            subdescription = os.path.basename(os.path.dirname(dicom_path))
                            uid_dict[uid] = (dicom_path, patient_id, description, subdescription, file)
                        except Exception as e:
                            print(f"Error processing file {dicom_path}: {e}")
    return dict(sorted(uid_dict.items(), key=lambda item: (item[1][1], item[1][4])))