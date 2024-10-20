import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from app import app
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import numpy as np
import os
from ultralytics import YOLO
import cv2
import requests
import torch
import tempfile
from YOLOv8_Explainer import yolov8_heatmap
import gdown


import os

# Model URL (Google Drive direct download link)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1DYqFQPWskUHUK0H1wFlt8mVdPHidnXBA"

# Download the model using gdown
def download_model(url, save_path):
    gdown.download(url, save_path, fuzzy=True)

# Local path to save the downloaded model
model_path = os.path.join(os.path.dirname(__file__), 'yolo-tumor-detection.pt')

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    download_model(MODEL_URL, model_path)

    # Print the download location and the file extension
    print(f"Model downloaded to: {model_path}")




# Load YOLOv8 model
model = YOLO(model_path)

# Dictionary containing label data
label_dict = {'1': ['2 0.502857 0.675714 0.851429 0.648571'], 'gg (28)': ['0 0.518555 0.470703 0.091797 0.082031', '0 0.427734 0.308594 0.070312 0.093750', '3 0.542969 0.604492 0.082031 0.091797'], 'gg (301)': ['0 0.550781 0.352539 0.070312 0.080078'], 'gg (4)': ['0 0.404297 0.289062 0.050781 0.078125', '0 0.482422 0.363281 0.089844 0.074219'], 'gg (460)': ['0 0.491211 0.276367 0.134766 0.103516', '0 0.583008 0.302734 0.033203 0.023438'], 'gg (618)': ['0 0.666016 0.540039 0.132812 0.169922'], 'gg (651)': ['0 0.640625 0.492188 0.144531 0.125000'], 'gg (70)': ['0 0.441406 0.258789 0.144531 0.197266'], 'gg (724)': ['0 0.617188 0.482422 0.191406 0.109375'], 'gg (763)': ['0 0.550781 0.348633 0.125000 0.263672'], 'image (10)': ['2 0.493333 0.357778 0.960000 0.697778'], 'image (30)': ['2 0.499023 0.500000 0.677734 0.878906'], 'image (58)': ['2 0.502119 0.448148 0.995763 0.792593'], 'image(155)': ['2 0.502119 0.343220 0.766949 0.576271'], 'image(255)': ['2 0.502262 0.502262 0.828054 0.995475'], 'image(289)': ['2 0.495882 0.391275 0.780000 0.707383'], 'image(295)': ['2 0.493056 0.498294 0.939815 0.894198'], 'image(298)': ['2 0.498667 0.514000 0.653333 0.798667'], 'm (105)': ['1 0.429688 0.375977 0.171875 0.119141'], 'm (3)': ['1 0.546875 0.402344 0.117188 0.128906'], 'm (53)': ['3 0.488281 0.449219 0.136719 0.125000', '3 0.486328 0.614258 0.117188 0.083984'], 'm1(106)': ['1 0.696907 0.522549 0.239175 0.241176'], 'm1(186)': ['1 0.681395 0.686364 0.311628 0.290909'], 'm1(47)': ['1 0.485352 0.414062 0.142578 0.156250'], 'm3 (25)': ['1 0.750000 0.312500 0.117188 0.121094'], 'p (12)': ['3 0.513672 0.386719 0.097656 0.074219'], 'p (146)': ['3 0.408203 0.562500 0.207031 0.281250'], 'p (484)': ['3 0.508789 0.543945 0.111328 0.154297'], 'p (49)': ['3 0.489258 0.566406 0.099609 0.101562'], 'p (75)': ['3 0.413086 0.501953 0.138672 0.179688'], 'p (778)': ['3 0.494141 0.370117 0.125000 0.119141'], 'Te-gl_0264': ['0 0.533203 0.509766 0.125000 0.117188'], 'Tr-me_0232': ['1 0.589367 0.263575 0.192308 0.142534'], 'Tr-me_0369': ['1 0.649414 0.371094 0.244141 0.222656'], 'Tr-me_0491': ['1 0.419922 0.511719 0.210938 0.187500'], 'Tr-no_0243': ['2 0.489683 0.398413 0.579365 0.542857'], 'Tr-pi_0115': ['3 0.401367 0.623047 0.064453 0.066406'], 'Tr-pi_0827': ['3 0.586914 0.506836 0.134766 0.185547'], 'Tr-pi_0838': ['3 0.593750 0.577148 0.132812 0.166016'], 'Tr-pi_0980': ['3 0.572266 0.584961 0.140625 0.126953']}

# Helper function to run YOLOv8 inference
def run_yolo_inference(image):
    image_copy = image.copy()
    results = model(image_copy)
    result_image = results[0].plot()
    return result_image

# Helper function to process the image for YOLOv8
def parse_image(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(BytesIO(decoded)).convert('RGB')
    return np.array(img)

# Helper function to encode the processed image back to base64 format
def encode_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

# Helper function to overlay YOLO labels on the original image using the dictionary
def overlay_labels_on_image(image, filename):
    base_name = os.path.splitext(filename)[0]
    
    # Check if labels for the image exist in the dictionary
    if base_name not in label_dict:
        return np.array(image)
    
    labels = label_dict[base_name]
    
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    # Iterate over the labels and draw them on the image
    for label in labels:
        parts = label.strip().split()
        class_id, x_center, y_center, width, height = map(float, parts)

        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = int(x_center - (width / 2))
        y1 = int(y_center - (height / 2))
        x2 = int(x_center + (width / 2))
        y2 = int(y_center + (height / 2))

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    return np.array(image)

# Generate LayerCAM heatmap using YOLOv8_Explainer
def generate_layercam_heatmap(image):
    image_copy = image.copy()
    temp_image_path = tempfile.NamedTemporaryFile(suffix=".png").name
    cv2.imwrite(temp_image_path, np.array(image_copy))

    device = torch.device("cpu")
    model_eigencam = yolov8_heatmap(
        weight=model_path,
        conf_threshold=0.4,
        device=device,
        method="LayerCAM",
        layer=[18],
        ratio=0.04,
        show_box=False,
        renormalize=False
    )

    heatmap_images = model_eigencam(img_path=temp_image_path)
    
    return heatmap_images[0]

# Layout for the Tumor Detection page
layout = dbc.Container(
    fluid=True,
    children=[
        html.Div(
            children=html.H2("Tumor Detection using YOLOv8", className="app-banner"),
            style={'margin-bottom': '30px'}
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Upload(
                            id='upload-image-tumor',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select an Image')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'color': '#007bff'
                            },
                            multiple=False
                        ),
                        html.Div(id='output-image-tumor', className="text-center", style={'padding-top': '20px'}),
                        html.Div(id='download-button-container-tumor', className="text-center mt-3")
                    ],
                    width=8,
                    style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}
                ),
                dbc.Col(
                    [
                        html.Div("Model Settings", className="text-center mb-3"),
                        html.H5("View Options", className="text-center"),
                        dcc.RadioItems(
                            id='view-option',
                            options=[
                                {'label': ' Original Image', 'value': 'original'},
                                {'label': ' Model Tumor Prediction', 'value': 'prediction'},
                                {'label': ' Original with Labels', 'value': 'original_labels'}
                            ],
                            value='original',
                            labelStyle={'display': 'block', 'margin-bottom': '10px'},
                            style={'padding': '20px 0'}
                        ),
                    ],
                    width=4,
                    style={'border-left': '1px solid #ddd', 'padding-left': '20px'}
                )
            ]
        )
    ]
)

# Single combined callback to handle both YOLO prediction and LayerCAM heatmap generation
@app.callback(
    [Output('output-image-tumor', 'children'),
     Output('download-button-container-tumor', 'children')],
    [Input('upload-image-tumor', 'contents'),
     Input('view-option', 'value')],
    [State('upload-image-tumor', 'filename')]
)
def update_output_and_layercam(contents, view_option, filename):
    if contents is not None:
        # Process the uploaded image
        img = parse_image(contents)
        pil_image = Image.open(BytesIO(base64.b64decode(contents.split(",")[1])))

        if view_option == 'original':
            image_to_display = img
        elif view_option == 'prediction':
            result_image = run_yolo_inference(img)
            image_to_display = result_image
        elif view_option == 'original_labels':
            image_to_display = overlay_labels_on_image(pil_image, filename)

        # Encode the selected image for display
        encoded_image = encode_image(image_to_display)

        # Create download button for YOLO image
        download_button = html.A(
            dbc.Button("Download Image", color="primary"),
            href=encoded_image,
            download=f"{filename}" if filename else "image.png"
        )

        # Generate LayerCAM heatmap
        layercam_image = generate_layercam_heatmap(img)

        # Ensure LayerCAM image is a NumPy array before fixing the colors
        if not isinstance(layercam_image, np.ndarray):
            layercam_image = np.array(layercam_image)

        # Fix the color channels (BGR to RGB)
        layercam_image_rgb = cv2.cvtColor(layercam_image, cv2.COLOR_BGR2RGB)

        # Encode LayerCAM heatmap for download
        encoded_heatmap_image = encode_image(layercam_image_rgb)

        # Create download button for LayerCAM heatmap
        download_heatmap_button = html.A(
            dbc.Button("Download LayerCAM heatmap", color="secondary"),
            href=encoded_heatmap_image,
            download=f"{filename}_layercam_heatmap.png" if filename else "image_layercam_heatmap.png"
        )

        # Return the original/prediction/label image and keep both download buttons available
        return [
            html.Img(src=encoded_image, style={'max-width': '100%', 'max-height': '80vh', 'display': 'block'}),
            html.Div([download_button, download_heatmap_button], className="d-flex justify-content-around")
        ]
    
    return ["Upload an image for prediction", ""]
