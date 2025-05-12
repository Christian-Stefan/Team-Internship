import numpy as np
import matplotlib.pyplot as plt
import json, glob, re, os, pydicom
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes,label,binary_dilation,binary_erosion, binary_closing


def get_slices(Json_path):

    '''
        Json_path = segmentation.json file path for the current scan

    '''

    file = glob.glob(Json_path)
    file = file[0].replace("\\", "/")
    slice_paths = []
    slice_coords = []

    with open(file) as f:
        d = json.load(f)
        dicom_file_base = re.sub(r"segmentation.*\.json", "1-", file)
        for slice in range(0,len(d["annotation"])):
            slice_path = dicom_file_base +str(next(iter(d["annotation"][slice]))).zfill(4)+".dcm"
            slice_paths.append(slice_path)
            slice_coord = d["annotation"][slice][next(iter(d["annotation"][slice]))]['segmentation'][0]
            slice_coords.append(slice_coord)
    return slice_paths, slice_coords


def preprocess(slice_paths,slice_coords,save_base_path,plot = False,save = False, segment = True,output = False):

    '''
        slice_paths: relative paths for each slice in a scan containing a nodule
        slice_coords: coordinates of nodule for each slice in a scan
        plot: want to plot the slices?
        save: want to save the slices?
        segment: want to output only nodule (so segmented) or the entire lungs?
        output: want to return the output?

    '''

    processed_out = []

    for slice_index in range(0, len(slice_coords)):
        slice = slice_coords[slice_index]
        slice_path = slice_paths[slice_index]

        dicom_data = pydicom.dcmread(slice_path)
        image_org = dicom_data.pixel_array

        image = dicom_data.pixel_array.astype(np.float32)

        image_out = segment_lungs(image)

        if sum(sum(image_out))>5000000:
            image_out = segment_lungs(image, threshold_non_black = 200, threshold_lung_mult = 0.4)

        if segment:
            image_out = segment_nodule(image=image,lungs_only=image_out,slice=slice)

        processed_out.append(image_out)

        if plot:
            fig, ax = plt.subplots()

            ax.imshow(image_out, cmap=plt.cm.bone, vmin=image_org.min(), vmax=image_org.max())
            #plt.title("Extracted Region from DICOM")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_position([0, 0, 1, 1])

        # Path to save the PNG file
        if save:
            ## Path to save the new DICOM
            output_dicom_path = save_base_path
            filename = str(slice_index)+".dcm"

            path = os.path.join(output_dicom_path, filename)

            # Save the padded image as a new DICOM file
            save_as_dicom(dicom_data, image_out, path)
    if output:
        return processed_out



def segment_lungs(image, threshold_non_black = 100, threshold_lung_mult = 1):
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

def segment_nodule(image,lungs_only,slice):
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

    cropped_image = pad_to_size(cropped_image,64)
    before_image = cropped_image.copy()
    med = np.median(cropped_image[cropped_image != 0])
    mask_fill = (cropped_image >= (med * 0.95)).astype(int)
    mask_fill = binary_closing(mask_fill, structure=np.ones((3, 3)))

    cropped_image_out = np.zeros_like(cropped_image)
    cropped_image_out[mask_fill.astype(bool)] = before_image[mask_fill.astype(bool)]

    return cropped_image_out


def pad_to_size(image,size):
    height, width = image.shape
    
    # Calculate padding amounts
    pad_top = (size - height) // 2 if (size - height) // 2 >=0 else 0
    pad_bottom = size - height - pad_top if size - height - pad_top >=0 else 0
    pad_left = (size - width) // 2 if (size - width) // 2 >=0 else 0
    pad_right = size - width - pad_left if size - width - pad_left >=0 else 0



    # Apply padding (black pixels = 0)
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return padded_image

def save_as_dicom(original_dicom, image_array, output_path):
    # Create a new DICOM dataset based on the original
    new_dicom = original_dicom.copy()

    # Update pixel data with the new padded image
    new_dicom.Rows, new_dicom.Columns = image_array.shape
    new_dicom.PixelData = image_array.astype(np.uint16).tobytes()  # Convert to bytes

    # Save as DICOM
    new_dicom.save_as(output_path)
    print(f"Saved: {output_path}")
