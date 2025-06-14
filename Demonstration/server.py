from flask import Flask, request, render_template, send_from_directory
from nodulereconstruction_v0_4 import Nodule_Reconstruction
from pipeline import Preprocessing, ModelDataInput
import os
import webbrowser
import threading
from model import ModelClassification
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('Demonstrator.html')

@app.route('/run', methods=['POST'])
def run_reconstruction():
    dcm_path = request.form['dcm_path']
    json_path = request.form['json_path']

    try:
        reconstructor = Nodule_Reconstruction(dcm_path, json_path)
        volume = reconstructor.build_3d_Mask()
        reconstructor.visualize_3d_Mask(volume)
        reconstructor.compute_max_nodule_extent(volume)

        run_pipeline(dcm_path, json_path)

        # Read result file
        prediction_text = ""
        if os.path.exists("static/result.txt"):
            with open("static/result.txt", "r") as f:
                prediction_text = f.read()

        return render_template(
            'Demonstrator.html',
            prediction_text=prediction_text
        )

    except Exception as e:
        return f"<h2>Error occurred:</h2><pre>{str(e)}</pre>"

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

def run_pipeline(dcm_path, json_path):
    try:
        dcm_path_nodule:str = f'Data/nodule'
        dcm_path_lung:str = 'Data/lung'

        Processing = Preprocessing(Root_path=dcm_path, Output_path=dcm_path_lung)
        dcm_collected_data = Processing.get_slice_and_coordnates_Paths()
        Processing.preprocess_Data(dcm_collected_data, segment=False)
        Processing = Preprocessing(Root_path=dcm_path, Output_path=dcm_path_nodule)
        Processing.preprocess_Data(dcm_collected_data, segment=True)

        # Load-in data for a prediction
        load_prediction(dcm_path_nodule, dcm_path_lung, json_path)

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")


def load_prediction(dcm_path_nodule, dcm_path_lung, json_path):

    try:
        token = json_path.split('\\')[-1].split('segmentation_0')[1].split('.json')[0]
        print(token)
        local_Nodule = ModelDataInput(json_path=json_path, scan_path=dcm_path_nodule + '/LIDC-IDRI-'+token, local_or_context="Local", naming_method="Pipeline")
        context_Lung = ModelDataInput(json_path=json_path, scan_path=dcm_path_lung + '/LIDC-IDRI-'+token, local_or_context="Context",naming_method="Pipeline")
        local_Nodule.get_radiomics()
        context_Lung.get_radiomics()

        # Make a prediction 
        make_prediction(local_Nodule, context_Lung, json_path)

    except Exception as e:
        print(f"[ERROR] ModelDataInput failed as {e}")

   
def make_prediction(local, context, json_path):
    try:
        ev = ModelClassification(
            modelpkl="triple_fusion_model_altdifflossv4.pkl",
            volume_local=local.volume_local,
            volume_context=context.volume_context,
            radiomics=local.radiomics,
            labels=local.labels
        )
        
        prediction_result = str(ev.get_result())
        print("PREDICTION",prediction_result)

        with open(json_path, 'r') as f:
            data = json.load(f)

        characteristics = data.get('characteristics', [])
        ground_truth = characteristics[0].get('nodule_name', 'Unknown')

        # Save to txt file
        os.makedirs("static", exist_ok=True)
        with open("static/result.txt", "w") as out_file:
            out_file.write(f"Prediction: {prediction_result}\n")
            out_file.write(f"Ground truth: {str(ground_truth)}\n")

        display_heat_map(local=local, context=context, ev=ev)

    except Exception as e:
        print(f"[ERROR] ModelClassification failed as {e}")


def display_heat_map(local, context, ev):

    def normalize(x):
        x = x - x.min()
        x = x / (x.max() + 1e-5)
        return x
    
    try:
        # General CFG
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_layer_lung = ev.model.context_branch.conv[4] # the second Conv3D layer
        target_layer_nodule = ev.model.local_branch.resblock1  # the second Conv3D layer

        def _display_save_lung(context_lung, target_layer_lung, ev_lung):
            input_tensor = torch.from_numpy(context_lung.volume_context).float().to(device)

            if input_tensor.ndim ==4:
                input_tensor = input_tensor.unsqueeze(0) # Add batch dimension

            target_category = context_lung.labels.item() if isinstance(context_lung.labels, torch.Tensor) else int(context_lung.labels.iloc[0])

            # Initialize Grad-CAM
            cam = GradCAM(model=ev_lung.model.context_branch, target_layers=[target_layer_lung])

            # Run Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])

            volume = context_lung.volume_context[0]
            grayscale = grayscale_cam[0]

            # Takes the slice in the middle
            mid = volume.shape[0] // 2
            slice_img = normalize(volume[mid])
            cam_slice = normalize(grayscale[mid])

            if cam_slice.shape != slice_img.shape:
                cam_slice = cv2.resize(cam_slice, slice_img.shape[::-1])

            # Convert CAM to heatmap
            cam_uint8 = np.ascontiguousarray((cam_slice * 255).astype(np.uint8))
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(0.7 * np.stack([slice_img]*3, axis=-1) + 0.3 * heatmap)
            ax.axis("off")

            # Save to static folder
            fig.tight_layout()
            fig.savefig("static/explainable_map_lungs.png", bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        def _display_save_nodule(local_nodule, target_layer_nodule, ev_nodule):

            # Convert numpy to tensor if needed and ensure dtype and device
            input_tensor = torch.from_numpy(local_nodule.volume_local).float().to(device)

            # Ensure it has shape [B, C, D, H, W]
            if input_tensor.ndim == 4:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

            # Process target label
            target_category = local_nodule.labels.item() if isinstance(local_nodule.labels, torch.Tensor) else int(local_nodule.labels.iloc[0])

            # Initialize Grad-CAM
            cam = GradCAM(model=ev_nodule.model.local_branch, target_layers=[target_layer_nodule])

            # Run Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])

            volume = local_nodule.volume_local[0]
            grayscale = grayscale_cam[0]                                    

            # Take the middle slice
            mid = volume.shape[0] // 2
            slice_img = normalize(volume[mid])
            cam_slice = normalize(grayscale[mid])

            if cam_slice.shape != slice_img.shape:
                cam_slice = cv2.resize(cam_slice, slice_img.shape[::-1])

            # Convert CAM to heatmap
            cam_uint8 = np.ascontiguousarray((cam_slice * 255).astype(np.uint8))
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

      
            # Prepare grayscale CT image
            ct_rgb = np.stack([slice_img] * 3, axis=-1)

            overlay = 0.8 * ct_rgb + 0.2 * heatmap

            plt.figure(figsize=(10, 6))
            plt.imshow(overlay)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("static/explainable_map_nodule.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        _display_save_lung(context, target_layer_lung, ev)
        _display_save_nodule(local_nodule=local, target_layer_nodule = target_layer_nodule, ev_nodule=ev)

    except Exception as e:
        print(f" {e}")

if __name__ == '__main__':
    port = 5000
    url = f'http://localhost:{port}'

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(debug=True, port=port)
