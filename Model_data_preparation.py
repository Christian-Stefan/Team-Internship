import os
import glob
import pydicom
import cv2
import numpy as np
import json
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
import re



# this class is just used for data exploration, and not necessary for the pipeline
class NoduleData:
    BENIGN_TYPES = {
            "Granuloma", "Active Infection", "Sarcoidosis", "Hamartoma",
            "Bronchioloalveolar Hyperplasia", "Long Covid", "Intrapulmonary Lymph Nodes"
        }

    MALIGNANT_TYPES = {
        "Adenocarcinoma", "Squamous Cell Carcinoma", "Large Cell (Undifferentiated) Carcinoma",
        "Small Cell Lung Cancer (SCLC)", "Carcinoid Tumors", "Sarcomatoid Carcinoma",
        "Lymphoma", "Adenoid Cystic Carcinoma", "Metastatic Tumors"
    }

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.json_paths = self._find_json_files()
        self.data = []

    def _find_json_files(self):
        patient_folders = [
            os.path.join(self.base_path, d)
            for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        ]
        json_paths = []
        for folder in patient_folders:
            json_files = glob.glob(os.path.join(folder, "*", "*", "*.json"))
            if json_files:
                json_paths.append(json_files[0])  # Assuming one JSON per patient
        return json_paths

    def _classify_nodule(self, name: str) -> str:
        if name in self.BENIGN_TYPES:
            return "Benign"
        elif name in self.MALIGNANT_TYPES:
            return "Malignant"
        else:
            return "Unknown"

    def extract_data(self):
        for file_path in self.json_paths:
            with open(file_path, "r") as f:
                data = json.load(f)

            characteristics = data.get("characteristics", [])
            slice_thickness = data.get("slice_thickness")
            annotations = data.get("annotation", [])

            slice_numbers = []
            for ann in annotations:
                if isinstance(ann, dict):
                    slice_numbers.extend(map(int, ann.keys()))
            slice_numbers = sorted(set(slice_numbers))

            for char in characteristics:
                name = char.get("nodule_name")
                self.data.append({
                    "nodule_name": name,
                    "nodule_category": self._classify_nodule(name),
                    "calcification": char.get("calcification"),
                    "internal_structure": char.get("internal_structure"),
                    "lobulation": char.get("lobulation"),
                    "margin": char.get("margin"),
                    "nodule_type": char.get("nodule_type"),
                    "sphericity": char.get("sphericity"),
                    "texture": char.get("texture"),
                    "global_seed": char.get("global_seed"),
                    "slice_thickness": slice_thickness,
                    "slices_present": slice_numbers,
                    "file_path": file_path
                })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def save_csv(self, output_path: str):
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

    def process_data(self):
        self["slices_present"] = self["slices_present"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        self = self[self['slices_present'].apply(lambda x: len(x) > 0)]
        
        le = LabelEncoder()
        self['label'] = le.fit_transform(self['nodule_name'])


# This class is used for formatting the input data for the model
class ModelDataInput:
    def __init__(self, json_path: str, scan_path:str, local_or_context:str, naming_method:str):
        self.json_path = json_path
        self.scan_path = scan_path
        self.local_or_context = local_or_context
        self.naming_method = naming_method
        self.data = []
        self.locals  = []
        self.contexts = []
        self.labels = []

    def get_radiomics(self):

        self.__extract_data()
        self.__to_dataframe()
        self.__process_data()

        match = re.search(r"LIDC-IDRI-\d{4}", self.scan_path)

        for slice_list in self.data["slices_present"]:
            for idx, slice_index in enumerate(slice_list):
                if "Local" in self.local_or_context:
                    if "Pipeline" in self.naming_method:
                        self.locals.append(self.__slice_load(dcm_path = self.scan_path +"\\"+ "Nodule"+str(match)+"_1-" + str(slice_index.zfill(4))+".dcm" , size=(64,64)))
                    
                    if "Aidan" in self.naming_method:
                        self.locals.append(self.__slice_load(dcm_path = self.scan_path +"\\"+ str(idx)+".dcm" , size=(64,64)))
                    
                if "Context" in self.local_or_context:
                    if "Pipeline" in self.naming_method:
                        self.contexts.append(self.__slice_load(dcm_path = self.scan_path +"\\"+ "Lung"+str(match)+"_1-" + str(slice_index.zfill(4))+".dcm",size=(256,256)))
                    if "Aidan" in self.naming_method:
                        self.contexts.append(self.__slice_load(dcm_path = self.scan_path +"\\"+ str(idx)+".dcm",size=(256,256)))

    def __extract_data(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)

        characteristics = data.get("characteristics", [])
        slice_thickness = data.get("slice_thickness")
        annotations = data.get("annotation", [])

        slice_numbers = []
        for ann in annotations:
            if isinstance(ann, dict):
                slice_numbers.extend(map(int, ann.keys()))
        slice_numbers = sorted(set(slice_numbers))

        for char in characteristics:
            name = char.get("nodule_name")
            self.data.append({
                "nodule_name": name,
                "calcification": char.get("calcification"),
                "internal_structure": char.get("internal_structure"),
                "lobulation": char.get("lobulation"),
                "margin": char.get("margin"),
                "nodule_type": char.get("nodule_type"),
                "sphericity": char.get("sphericity"),
                "texture": char.get("texture"),
                "slice_thickness": slice_thickness,
                "slices_present": slice_numbers,
            })
        
    def __to_dataframe(self):
        self.data = pd.DataFrame(self.data)
    
    def __process_data(self):
        self.data["slices_present"] = self.data["slices_present"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        self.data = self.data[self.data['slices_present'].apply(lambda x: len(x) > 0)]
        
        le = LabelEncoder()
        self.data['label'] = le.fit_transform(self.data['nodule_name'])
        self.labels = self.data['label']

    def __slice_load(self,dcm_path, size):
        dcm_image = pydicom.dcmread(dcm_path)
        img = dcm_image.pixel_array

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)

        # Pad if needed
        cropped_image = cv2.resize(img, size)

        return cropped_image
