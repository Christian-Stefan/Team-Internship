# Team-Internship

This branch contains the code for the preprocessing pipeline and is what the preprocessing class in the Chris branch is based on.

## preprocessing.py

This file contains two steps: 
- first a method to get all the file paths of the dicom slices that contain each nodule, and also gets the coordinates of the nodules in those slices.
- second a method to segment the lungs or the nodules from those slices and return these as a dicom file.

Getting the relevant file paths and coordinates is based on the json file connected to every scan.

To segment the lungs, first the area which envelops the human body in the slice is found with a binary mask, then to only get the lungs the Otsu threshold is used. Then with some binary dilation and filling holes inside the area only the lungs remain. After this binary erosion is needed to remove the border around the lungs.

To get only the nodule back, first the area is taken based on the coordinates from the json. However, these coordinates are slightly off from the actual location of the nodule in a random way. So the taken area needs to be expanded in all direction by some pixels with binary dilation. This mask is then applied to the segmented lungs. Then the image is cropped. After this the median of all pixel values is taken to remove all background pixels that are not the nodule.

## segment.ipynb

this notebook was used to generate all the segmented dicom files and save them.

The Local context branch notebooks use the same environment as this branch and the output of the pre-processed dicom files from segment.ipynb
