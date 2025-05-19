Documentation 
==========

Nodule_Reconstruction
----------------------

This repository contains a class `Nodule_Reconstruction` designed for reconstructing and visualizing 3D lung nodules from medical imaging segmentation files in JSON format. It enables researchers or developers to:
- Generate a 3D mask from the segmentation data.
- Visualize the resulting volume.
- Compute the maximum spatial extent of the nodule.

Dependencies
-------------
A `requirements.txt` file is **not** included in this repository.

Before using the class, please ensure the following Python packages are installed in your environment:
- numpy
- matplotlib
- plotly
- nibabel
- scikit-image
- json
- os

You can manually install them using pip:

Usage
------
Below is an example of how to use the class in a Python script or notebook:

```python
from Nodule_Reconstruction import Nodule_Reconstruction

# Initialize with folder and segmentation path
nodule = Nodule_Reconstruction('/content/sample_data',
                               '/content/sample_data/segmentation_00037.json')

# Generate the 3D mask volume from the segmentation
volume = nodule.build_3d_Mask()

# Visualize the reconstructed nodule in 3D
nodule.visualize_3d_Mask(volume)

# Compute the maximum extent (size) of the nodule in 3D space
nodule.compute_max_nodule_extent(volume)
````


🩺 Lung & Nodule CT Preprocessing Pipeline
-------------------------------------------

This repository provides a DICOM preprocessing pipeline for CT scans, including:
- Lung segmentation
- Nodule segmentation using annotated JSON files
- Slice-by-slice visualization
- Exporting processed images as new DICOM files

---

## 📂 Folder Structure

The code assumes the following folder structure under your root path:

```
Input_Data/
├── LIDC-IDRI-0637/
│   └── 01-01-2000-NA-NA-64428/
│       └── 187215.000000-NA-14556/
│           └── 1-0024.dcm
├── LIDC-IDRI-0644/
│   └── 01-01-2000-NA-NA-65107/
│       └── 936488.000000-NA-30457/
│           ├── 1-0001.dcm
│           ├── 1-0002.dcm
│           └── 1-0003.dcm
```

Each scan folder includes:
- Multiple `.dcm` files (slices)
- A single `.json` annotation file that defines nodule coordinates

---

## ⚙️ Class Overview

### `Preprocessing`

```python
Preprocessing(Output_path: str, Root_path: str, Json_path: str = None)
```

- `Output_path`: Directory to save processed DICOMs
- `Root_path`: Directory containing raw DICOM data
- `Json_path`: Optional – not used directly in current implementation - Prefferably is not to be used

---

## 🚀 How to Use

### 1. Initialize Preprocessing
```python
Processing = Preprocessing(
    Output_path="/path/to/output",
    Root_path="/path/to/input"
)
```

### 2. Collect Paths and Coordinates
```python
dcm_collected_data = Processing.get_slice_and_coordnates_Paths()
```
This will:
- Traverse the input directory
- Find folders containing both `.dcm` and `.json`
- Parse the `.json` for slice/nodule coordinates
- Return a list of `[slice_paths + json_path, nodule_coordinates]`

### 3. Preprocess and Save

#### 🔸 Segment Nodules
```python
Processing.preprocess_Data(
    data=dcm_collected_data,
    plot=True,       # Visualize a random sample
    save=True,       # Save as DICOMs
    segment=True     # Segment nodules
)
```

#### 🔹 Segment Lungs (without nodules)
```python
Processing.preprocess_Data(
    data=dcm_collected_data,
    plot=True,
    save=True,
    segment=False     # Segment only lungs
)
```

### 💡 Notes:
- Currently, lungs and nodules must be processed separately.
- The saved filenames follow this pattern:  
  - `NoduleLIDC-IDRI-0010_1-0054.dcm`  
  - `LungLIDC-IDRI-0010_1-0054.dcm`

---

## 🖼 Visualization

If `plot=True`, a 3D scrollable view of the preprocessed slices will be shown using `ipywidgets`.

### 3D Volume Explorer
```python
Processing.explore_3D_array(volume_z_first)
```

### Before/After Comparison
```python
Processing.explore_3D_array_comparison(arr_before, arr_after)
```

---

## 💾 Saved Files

Saved DICOMs are exported into the path provided in `Output_path`:
- All files go into a **flat directory** (no folder structure is preserved)
- Filenames include the CT ID and slice ID for traceability

---
