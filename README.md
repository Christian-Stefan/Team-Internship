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
