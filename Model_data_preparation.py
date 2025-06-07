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
        self.df = pd.DataFrame()
        self.locals  = []
        self.contexts = []
        self.volume_local = []
        self.volume_context = []
        self.labels = []
        self.radiomics = []
        self.le = LabelEncoder()

    def get_radiomics(self):

        self.__extract_data()
        self.__to_dataframe()
        self.__process_data()
        self.__get_radiomics()

        match = re.search(r"LIDC-IDRI-\d{4}", self.scan_path)

        for slice_list in self.df["slices_present"]:
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
        if "Local" in self.local_or_context:
            self.volume_local = self.__pad_or_crop_volume(self.locals,target_depth=5, shape=(64, 64))
        if "Context" in self.local_or_context:
            self.volume_context = self.__pad_or_crop_volume(self.contexts, target_depth=7, shape=(256, 256))
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
        self.df = pd.DataFrame(self.data)
    
    def __process_data(self):
        self.df["nodule_name"] = self.df["nodule_name"].replace("Hyperplasia", "Bronchioloalveolar Hyperplasia")
        self.df["nodule_name"] = self.df["nodule_name"].replace("Granuloma - Active Infection", "Active Infection")
        self.df["slices_present"] = self.df["slices_present"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        self.df = self.df[self.df['slices_present'].apply(lambda x: len(x) > 0)]
        
        class_names = [
        "Active Infection",
        "Adenocarcinoma",
        "Adenoid Cystic Carcinoma",
        "Bronchioloalveolar Hyperplasia",
        "Carcinoid Tumors",
        "Granuloma",
        "Hamartoma",
        "Intrapulmonary Lymph Nodes",
        "Large Cell (Undifferentiated) Carcinoma",
        "Lymphoma",
        "Metastatic Tumors",
        "Sarcoidosis",
        "Sarcomatoid Carcinoma",
        "Small Cell Lung Cancer (SCLC)",
        "Squamous Cell Carcinoma",
        ]

        # Apply mapping
        self.df['label'] = self.df['nodule_name'].apply(lambda name: class_names.index(name) if name in class_names else -1)

        # Save the class list (optional)
        self.labels = self.df['label']

    def __slice_load(self,dcm_path, size):
        dcm_image = pydicom.dcmread(dcm_path)
        img = dcm_image.pixel_array

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)

        # Pad if needed
        cropped_image = cv2.resize(img, size)

        return cropped_image
    
    def __pad_or_crop_volume(self, slices,target_depth, shape):
        D = len(slices)
        h, w = shape

        if D == 0:
            return np.zeros((1, target_depth, h, w), dtype=np.float32)

        volume = np.stack(slices, axis=0)

        if D < target_depth:
            pad_before = (target_depth - D) // 2
            pad_after = target_depth - D - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
        elif D > target_depth:
            start = (D - target_depth) // 2
            volume = volume[start:start+target_depth]

        return volume[None, ...]  # Add channel dim: [1, D, H, W]
    
    def __get_radiomics(self):
        radiomics_columns = [
        "calcification", "internal_structure", "lobulation",
        "margin", "sphericity", "texture"
        ]
        one_hot_dict = {
            "calcification": [
                "calcification_Absent",
                "calcification_Central",
                "calcification_Laminated",
                "calcification_Non-Central",
                "calcification_Popcorn",
                "calcification_Solid",
            ],
            "internal_structure": [
                "internal_structure_Soft Tissue",
            ],
            "lobulation": [
                "lobulation_Marked",
                "lobulation_N-Marked",
                "lobulation_Nn-Mk",
                "lobulation_None-M",
            ],
            "margin": [
                "margin_P-Sharp",
                "margin_Poo-Sh",
                "margin_Poorly",
                "margin_Poorly-S",
                "margin_Sharp",
            ],
            "sphericity": [
                "sphericity_Lin-Ov",
                "sphericity_Linear",
                "sphericity_Ov-Ro",
                "sphericity_Ovoid",
                "sphericity_Round",
            ],
            "texture": [
                "texture_NS-PS",
                "texture_PS-Solid",
                "texture_Part Solid/Mixed",
                "texture_Solid",
            ],
        }

        # Extract just the columns of interest from self.df
        radiomics_df = self.df[radiomics_columns].copy()

        # One-hot encode these columns
        radiomics_encoded = pd.get_dummies(radiomics_df, drop_first=False)

        # Create a full list of expected columns by flattening the dict values
        full_columns = []
        for cols in one_hot_dict.values():
            full_columns.extend(cols)

        # Reindex to the full set of columns, filling missing columns with 0
        radiomics_encoded = radiomics_encoded.reindex(columns=full_columns, fill_value=0)

        # Convert to numpy array
        self.radiomics = radiomics_encoded.to_numpy(dtype=np.float32)