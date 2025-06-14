import SimpleITK as sitk
import time as t
import numpy as np
import matplotlib.pyplot as plt
import json, glob, re, os, pydicom
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes,label,binary_dilation,binary_erosion, binary_closing
from ipywidgets import interact
import shutil
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import ast

class Preprocessing:

  def __init__(self,
               Output_path:str,
               Root_path:str,
               Json_path:str = None):

    self.Json_path = Json_path
    self.Output_path = os.path.normpath(Output_path)
    self.Root_path = os.path.abspath(os.path.normpath(Root_path))

  def get_slice_and_coordnates_Paths(self,
                                     root_path:str=None):

    ct_collections:list = [] # CT container
    #labels:list = [] # Labels container

    if root_path is None:
        root_path = self.Root_path
    else:
        root_path = os.path.abspath(os.path.normpath(root_path))

    for dirpath, dirnames, filenames in os.walk(root_path):
        folder_name = os.path.basename(dirpath)
  
        if "-NA-" in folder_name:
            # Collect and sort full paths to .dcm files
            dcm_files = sorted(
                [os.path.join(dirpath, f) for f in filenames if f.lower().endswith('.dcm')]
            )

            # Identify the .json file
            json_files = [f for f in filenames if f.lower().endswith('.json')]
            json_file = json_files[0] if json_files else None

            if dcm_files and json_file:
                json_path = os.path.join(dirpath, json_file)

                # Load and parse JSON to extract annotations
                with open(json_path, 'r') as f:
                    d = json.load(f)

                slice_coords:list = []
                for annotation in d.get("annotation", []):
                    slice_key = next(iter(annotation))
                    coords = annotation[slice_key]['segmentation'][0]
                    slice_coords.append(coords)

                # Full list of paths: DICOMs + JSON
                full_sequence = dcm_files + [json_path]

                # Append both: [paths, coordinates]
                ct_collections.append([full_sequence, slice_coords])

    return ct_collections
  
  def preprocess_Data(self,
                data:list,
                plot:bool = False,
                save:bool = True,
                segment:bool = True,
                output:bool = True):

      '''
          slice_paths: relative paths for each slice in a scan containing a nodule
          slice_coords: coordinates of nodule for each slice in a scan
          plot: want to plot the slices?
          save: want to save the slices?
          segment: want to output only nodule (so segmented) or the entire lungs?
          output: want to return the output?

      '''
      # ---- Containers Definition ----- # Starts
      slices_coords:list = [] # Container with nodule cordinates
      slices_paths:list = [] # Container with nodule paths
      slices_z:list = [] # Deepness of each nodule belonging to a given CT
      slies_deepness:list = [] # Deepness of each CT
      slices_deepness:list = [len(data[deep][0]) for deep in range(len(data))] # Deepness of each CT through list comprehension
      nodule_labels:list = [] # List which is meant to contain labels, defining nodule category (e.g., nodule_name)
      processed_out:list = []
      index_coords:int = 0
      # ---- Containers Definition ----- # Ends

      # 1. Extracting relaive paths and nodule coordinates
      for file in data:
        if 'segmentation' in str(file[:][0]): # Sorting out coordinates and identifying all the paths
          json_file = file[:][0][(len(file[:][0])-1)] # 1.1 Selecting the very last element in the subdirectory which usually is .json
          with open(json_file,'r' ) as file_json:
            json_segmentation = json.load(file_json)
            for z, slices in enumerate(json_segmentation['annotation']):# 3.1. Extracting the number of slices (e.g., z axis)
              for slice in slices: # 3.2. Using the number indicating the exact slice to access the (x,y) coordinates
                # 2. After slices of interest have been localized dcmread is possible
                slices_coords.append([slices[slice]['segmentation'][0]]) # Appending Slices Coordinates
                if len(str(slice)) == 2:
                  expected_filename = f"1-{int(slice):04d}.dcm"
                  for slice_path in file[:][0]:
                    if expected_filename in slice_path:
                      slices_paths.append(slice_path) # Appending Slice Paths
            slices_z.append([z])
        # -- Baseline 1 -> slice_paths, slices_coords, slices_z,  nodule_laebels
      # 2. Parse .dicom dataset
      for ct in range(len(data)):

        processed_out_temporary:list = [] # Initialize a temporary container/chunk per CT

        ct_overall_deepness = len(data[ct][0]) # Parse the slice sequence of all CTs
        # print("Processing CT:{}, which has the deepness of {}\n".format(data[ct][0],
        #                                                               ct_overall_deepness))
        # 2.1. Read with .dcmread each slice
        for slice_index in range(slices_z[ct][0]+1):

          path = slices_paths[0]
          # print(f"Reading slice {slice_index} located at path...{path}") Debugging print
          dicom_data = pydicom.dcmread(path) # Reading slice by slice a given CT
          IMAGE_RAW = dicom_data.pixel_array # Raw image
          IMAGE_NOT_RAW = dicom_data.pixel_array.astype(np.float32) # Not-raw image

          # 3. Segment lung
          IMAGE_LUNG = self.segment_lungs(IMAGE_NOT_RAW)

          # 3.1 Differentiating thick zones in-between nodule corpus and edges of lung
          if sum(sum(IMAGE_LUNG))>5000000:
            IMAGE_LUNG = self.segment_lungs(IMAGE_NOT_RAW,
                                            threshold_non_black = 200,
                                            threshold_lung_mult = 0.4)
          # 4. Segment nodule
          if segment:
            IMAGE_LUNG = self.segment_nodule(IMAGE_NOT_RAW,
                                       IMAGE_LUNG,
                                       slice=slices_coords[index_coords][0])
            index_coords+=1 # Update nodule coordinate index

          if save:
              match = re.search(r"(LIDC-IDRI-\d{4})", path)  # Match LIDC-IDRI-xxxx
              filename1 = os.path.basename(path)  

              if match:
                folder_name = match.group(1)
              else:
                raise ValueError(f"Could not extract folder name from path: {path}")

              # Set filename prefix
              filename_prefix = 'Nodule' if segment else 'Lung'

              # Final filename
              filename = f"{filename_prefix}{folder_name}_{filename1}"  # Ex: LungLIDC-IDRI-0000_1-0053.dcm

              # Output directory
              output_dir = os.path.join(self.Output_path, folder_name)
              os.makedirs(output_dir, exist_ok=True)

              # Final full save path
              output_path = os.path.join(output_dir, filename)

              # Save image as DICOM
              self.save_as_dicom(dicom_data, IMAGE_LUNG, output_path)


              # Try to locate and copy the matching .json file from `data`
              folder_id = folder_name.split("-")[-1]  # e.g. "0637" from "LIDC-IDRI-0637"

              for group in range(len(data)):
                  entries = data[group]
                  for i in range(0, len(entries), 2):  # Step by 2
                      json_path = entries[i][-1]  # Full path to the .json
                      if f"_0{folder_id}.json" in json_path:
                            try:
                                json_filename = os.path.basename(json_path)
                                dest_path = os.path.join(output_dir, json_filename)
                                shutil.copy2(json_path, dest_path)
                            except Exception as e:
                                print(f"❌ Failed to copy JSON {json_path}: {e}")
                            break  # Stop after first match

          processed_out_temporary.append(IMAGE_LUNG)   # Adding processed slices in a temporary list-type chunk
          slices_paths.pop(0) # Removing the slice path from slices paths

        processed_out.append(processed_out_temporary) # Adding processed CTs in processed_out container

      # print(f"-------------------------------Number of CTs preprocessed-------------------------------: {len(processed_out)}\n") # Debugging print

      if plot:

        # 1. Visualise the preprocessed CT with most slices
        # Displaying the preprocessed CT with the most slices
        # Determining the deepness
        max_z_index = slices_z.index(max(slices_z)) # Identifying the CT's index with higest number of slices that contain nodule(s)
        len_z_on_index = len(processed_out[max_z_index]) # Determining the number of slices containing nodule(s)
        slices = processed_out[max_z_index]
        volume = np.stack(processed_out[max_z_index], axis=2)
        volume_z_first = np.transpose(volume, (2, 0, 1))
        self.explore_3D_array(volume_z_first)

        # 2. Visualise all the preprocesed CT in series
        # for group_index in range(len(slices_z)):
        #   volume = np.stack(processed_out[group_index], axis=2)
        #   volume_z_first = np.transpose(volume, (2, 0, 1))
        #   self.explore_3D_array(volume_z_first)

  def segment_lungs(self,
                    image,
                    threshold_non_black = 100,
                    threshold_lung_mult = 1):
    # Remove completely black background
    non_black_mask = image > threshold_non_black  # Ignore pure black regions outside scan
    scan_region = binary_fill_holes(non_black_mask)  # Fill holes in the scan area

    # Apply thresholding **only inside the scan region**
    thresh = threshold_otsu(image[scan_region])  # Compute threshold ignoring black border
    lung_mask = (image < thresh*threshold_lung_mult) & scan_region  # Select dark regions **inside the scan**

    # Morphological operations to clean up
    lung_mask = binary_dilation(lung_mask, iterations=3) # add back in some of the area around the lungs
    lung_mask = binary_fill_holes(lung_mask)  # Fill small holes, helps if nodule was close to lung wall

    # keep the two largest regions (lungs)
    labeled_mask, num_features = label(lung_mask)
    unique, counts = np.unique(labeled_mask, return_counts=True)
    sorted_labels = sorted(zip(unique[1:], counts[1:]), key=lambda x: -x[1])[:2]  # Top 2 largest
    lung_mask = np.isin(labeled_mask, [s[0] for s in sorted_labels])  # Keep only lungs

    lung_mask = binary_erosion(lung_mask, disk(4)) # remove lung wall

    # final mask
    lungs_only = np.zeros_like(image)
    lungs_only[lung_mask] = image[lung_mask]

    return lungs_only

  def segment_nodule(self,
                     image,
                     lungs_only,
                     slice):

    height, width = image.shape

    x_coords = slice[::2]
    y_coords = slice[1::2]

    x_coords_new = (np.array(x_coords) * width).astype(int)
    y_coords_new = (np.array(y_coords) * height).astype(int)


    mask = np.zeros_like(image, dtype=bool)

    # Get polygon fill area
    rr, cc = polygon(y_coords_new, x_coords_new, mask.shape)
    mask[rr, cc] = True  # Fill the polygon

    # Fill any holes inside the polygon
    mask = binary_fill_holes(mask)

    # Expand mask by 4 pixels
    mask = binary_dilation(mask, iterations=3)


    # Create the masked image
    masked_image = np.zeros_like(lungs_only)
    masked_image[mask] = lungs_only[mask]  # Retain original pixels inside the boundary

    min_x = min(x_coords_new)
    max_x = max(x_coords_new)
    min_y = min(y_coords_new)
    max_y = max(y_coords_new)
    padding = 5  # You can adjust this value as needed

    # Adjust the bounding box coordinates with padding
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    cropped_image = masked_image[min_y:max_y, min_x:max_x]

    cropped_image = self.pad_to_size(cropped_image,64)
    before_image = cropped_image.copy()
    med = np.median(cropped_image[cropped_image != 0])
    mask_fill = (cropped_image >= (med * 0.95)).astype(int)
    mask_fill = binary_closing(mask_fill, structure=np.ones((3, 3)))

    cropped_image_out = np.zeros_like(cropped_image)
    cropped_image_out[mask_fill.astype(bool)] = before_image[mask_fill.astype(bool)]

    return cropped_image_out

  def pad_to_size(self,
                  image,
                  size):
    height, width = image.shape

    # Calculate padding amounts
    pad_top = (size - height) // 2 if (size - height) // 2 >=0 else 0
    pad_bottom = size - height - pad_top if size - height - pad_top >=0 else 0
    pad_left = (size - width) // 2 if (size - width) // 2 >=0 else 0
    pad_right = size - width - pad_left if size - width - pad_left >=0 else 0



    # Apply padding (black pixels = 0)
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return padded_image


  def explore_3D_array(self,
                       arr: np.ndarray,
                       cmap: str = 'gray'):
    """
    Given a 3D array with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D array.
    The purpose of this function to visual inspect the 2D arrays in the image.

    Args:
      arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
      cmap : Which color map use to plot the slices in matplotlib.pyplot
    """

    def fn(SLICE):
      plt.figure(figsize=(12,12))
      plt.imshow(arr[SLICE, :, :], cmap=cmap)
      plt.show()

    interact(fn, SLICE=(0, arr.shape[0]-1))


  def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):
    """
    Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
    The purpose of this function to visual compare the 2D arrays after some transformation.

    Args:
      arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
      arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform
      cmap : Which color map use to plot the slices in matplotlib.pyplot
    """

    assert arr_after.shape == arr_before.shape

    def fn(SLICE):
      fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,10))

      ax1.set_title('Before', fontsize=15)
      ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)

      ax2.set_title('After', fontsize=15)
      ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)

      plt.tight_layout()
      plt.show()

    interact(fn, SLICE=(0, arr_before.shape[0]-1))

  def save_as_dicom(self,
                    original_dicom,
                    image_array,
                    output_path):
      # Create a new DICOM dataset based on the original
      new_dicom = original_dicom.copy()

      # Update pixel data with the new padded image
      new_dicom.Rows, new_dicom.Columns = image_array.shape
      new_dicom.PixelData = image_array.astype(np.uint16).tobytes()  # Convert to bytes

      # Save as DICOM
      new_dicom.save_as(output_path)
      #print(f"Saved: {output_path}")

class pipeline_norm_Extension(Dataset):

    def __init__(self, path_to_preprocessed_data):
        self.data_container: list = []
        self.Output_path = path_to_preprocessed_data
        self.label2idx: dict = {}

        # Structure became obosolete since Preprocessing is executed sepparatedly

        # # 1. Run preprocessing
        # dcm_collected_data = self.get_slice_and_coordnates_Paths()
        # self.preprocess_Data(dcm_collected_data, plot=False, save=True, segment=False)

        label_set = set()

        # 2. Load data
        for folder_name in os.listdir(self.Output_path):
            folder_path = os.path.join(self.Output_path, folder_name)
            if os.path.isdir(folder_path) and folder_name.startswith("LIDC-IDRI"):
                print(f"📁 Scanning folder: {folder_name}")
                label = ''  # reset label

                # Loop through DICOMs for segmentation...json
                for file in os.listdir(folder_path):
                    if 'segmentation' in file:
                        with open(os.path.join(folder_path, file), 'r') as f:
                            json_annotation = json.load(f)
                        label = json_annotation['characteristics'][0]['nodule_name']
                        label_set.add(label)
                        break

                # Loop through DICOMs for SLICES/LUNGS..dcm
                for file in os.listdir(folder_path):
                    if 'LIDC-IDRI' in file and file.endswith('.dcm'):
                        dicom_files = pydicom.dcmread(os.path.join(folder_path, file))
                        img_context = dicom_files.pixel_array.astype(np.int32)

                        # HU conversion
                        intercept = dicom_files.RescaleIntercept
                        slope = dicom_files.RescaleSlope
                        hu_image = slope * img_context + intercept

                        # Normalize
                        hu_image = np.clip(hu_image, -1000, 400)
                        normalized = (hu_image + 1000) / 1400.0
                        resized = cv2.resize(normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
                        channeled_image = np.expand_dims(resized, axis=-1).astype(np.float32)

                        # Add sample + label
                        if label != '':
                            prepared_sample = {'img': channeled_image, 'label': label}
                            self.data_container.append(prepared_sample)

        # Finalize label-to-index mapping
        self.label2idx = {label: idx for idx, label in enumerate(sorted(label_set))}

    def __len__(self):
        """
        Custom function that builts-in a `len` feature for our current class to determine
        the length of data_container container returned by the constructor

        """
        return len(self.data_container)

    def __getitem__(self, idx):
      """
      Each time an item is retrieved from an object/instance of this class it will pass through the bellow
      cascade of actions to make a tensor out of it eventually;

      """
      item = self.data_container[idx]
      img = torch.from_numpy(item['img']).permute(2, 0, 1).float()  # shape: (1, 224, 224)
      label = torch.tensor(self.label2idx[item['label']], dtype=torch.long)

      return img, label

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augment=True):
        self.base_dataset = base_dataset
        self.augment = augment

        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.augment:
            img = transforms.ToPILImage()(img)
            img = self.aug_transforms(img)
            img = transforms.ToTensor()(img)
        return img, label
    
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
        match = match.group()

        for slice_list in self.df["slices_present"]:
            for idx, slice_index in enumerate(slice_list):
                if "Local" in self.local_or_context:
                    if "Pipeline" in self.naming_method:
                        self.locals.append(self.__slice_load(dcm_path = self.scan_path +"\\"+ "Nodule"+str(match)+"_1-" + str(slice_index).zfill(4)+".dcm" , size=(64,64)))
                    
                    if "Aidan" in self.naming_method:
                        self.locals.append(self.__slice_load(dcm_path = self.scan_path +"\\"+ str(idx)+".dcm" , size=(64,64)))
                    
                if "Context" in self.local_or_context:
                    if "Pipeline" in self.naming_method:
                        self.contexts.append(self.__slice_load(dcm_path = self.scan_path +"\\"+ "Lung"+str(match)+"_1-" + str(slice_index).zfill(4)+".dcm",size=(256,256)))

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