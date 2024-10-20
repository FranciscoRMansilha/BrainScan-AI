import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from app import app
from PIL import Image, ImageEnhance
from io import BytesIO
import base64
import numpy as np
import scipy.ndimage as ndi
import cv2
import plotly.express as px

# Helper function to apply gamma correction
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table).astype("uint8")

    if image.mode == 'RGB':
        r, g, b = image.split()
        r = r.point(table)
        g = g.point(table)
        b = b.point(table)
        return Image.merge('RGB', (r, g, b))
    elif image.mode in ('L', 'P'):  # Grayscale
        return image.point(table)

    return image

# Helper function to apply CLAHE
def apply_clahe(image):
    img_gray = image.convert('L')  # Convert to grayscale for CLAHE
    img_np = np.array(img_gray)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_np)
    
    return Image.fromarray(clahe_img)

# Helper function to process the image and apply adjustments
def parse_contents(contents, brightness=1.0, contrast=1.0, gamma=1.0, clahe_applied=False, rotation=0):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(BytesIO(decoded))

    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img_enhanced = enhancer.enhance(brightness)

    # Apply contrast adjustment
    enhancer_contrast = ImageEnhance.Contrast(img_enhanced)
    img_contrasted = enhancer_contrast.enhance(contrast)

    # Apply gamma correction
    img_gamma_corrected = adjust_gamma(img_contrasted, gamma)

    # Apply CLAHE if enabled
    if clahe_applied:
        img_gamma_corrected = apply_clahe(img_gamma_corrected)

    # Convert image to numpy array for rotation
    img_np = np.array(img_gamma_corrected)
    
    # Apply rotation using scipy
    rotated_image = ndi.rotate(img_np, angle=rotation, reshape=False)

    return rotated_image  # Return numpy array of the image for zoom

# Helper function to encode image as base64
def encode_image(image_array):
    final_image = Image.fromarray(image_array)

    # Save the image to a buffer without changing its resolution
    buffer = BytesIO()
    final_image.save(buffer, format="PNG", quality=100)  # Save with maximum quality
    buffer.seek(0)

    # Encode the image back into base64 format
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

    return f"data:image/png;base64,{encoded_image}"

# Layout for the Image Manipulation page
layout = dbc.Container(
    fluid=True,
    children=[
        html.Div(
            children=html.H2("Image Analysis Toolkit", className="app-banner"),
            style={'margin-bottom': '30px'}
        ),
        dbc.Row(
            [
                # Left column for the image display and title
                dbc.Col(
                    [
                        # Image upload and display area
                        dcc.Upload(
                            id='upload-image',
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
                        # Display uploaded image with zoom functionality
                        dcc.Graph(id='image-zoom-graph', config={'scrollZoom': True}, style={'display': 'none'}),
                        # Centered download button below the image
                        html.Div(id='download-button-container', className="text-center mt-3")
                    ],
                    width=8,  # Take up 8 out of 12 columns for the left side
                    style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}
                ),
                # Right column for sliders and widgets
                dbc.Col(
                    [
                        # Reset Button
                        dbc.Button(
                            "Restore Defaults",
                            id='reset-button',
                            color="info",
                            className="mb-4",
                            style={'width': '100%', 'background-color': '#17a2b8'}
                        ),
                        html.H5("Adjust Brightness", className="text-center"),
                        dcc.Slider(
                            id='brightness-slider',
                            min=0.1,
                            max=2.0,
                            step=0.1,
                            value=1.0,  # Default value
                            marks={i: f"{i}" for i in range(0, 3)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.H5("Adjust Contrast", className="text-center mt-3"),
                        dcc.Slider(
                            id='contrast-slider',
                            min=0.1,
                            max=2.0,
                            step=0.1,
                            value=1.0,  # Default value
                            marks={i: f"{i}" for i in range(0, 3)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.H5("Adjust Gamma", className="text-center mt-3"),
                        dcc.Slider(
                            id='gamma-slider',
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            value=1.0,  # Default value
                            marks={i: f"{i}" for i in range(0, 4)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.H5("Rotate Image", className="text-center mt-3"),
                        dcc.Slider(
                            id='rotation-slider',
                            min=-180,
                            max=180,
                            step=1,
                            value=0,  # Default rotation is 0 degrees
                            marks={i: f"{i}°" for i in range(-180, 181, 45)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(
                            [
                                html.Span("Enhance Contrast (CLAHE)", className="switch-label"),
                                dbc.Switch(
                                    id='clahe-toggle',
                                    value=False,  # Default is no CLAHE applied
                                    className="switch-control"  # Use custom class for size control
                                )
                            ],
                            className="d-flex align-items-center justify-content-center mt-3"
                        )
                    ],
                    width=4,  # Take up 4 out of 12 columns for the right side
                    style={'border-left': '1px solid #ddd', 'padding-left': '20px'}
                )
            ]
        )
    ]
)

# Callback to update the output div with the image, handle brightness, contrast, gamma, rotation, and CLAHE adjustments, and show the download button
@app.callback(
    [Output('image-zoom-graph', 'figure'),
     Output('image-zoom-graph', 'style'),
     Output('download-button-container', 'children')],
    [Input('upload-image', 'contents'),
     Input('brightness-slider', 'value'),
     Input('contrast-slider', 'value'),
     Input('gamma-slider', 'value'),
     Input('rotation-slider', 'value'),
     Input('clahe-toggle', 'value')],
    [State('upload-image', 'filename')]
)
def update_output_image(contents, brightness, contrast, gamma, rotation, clahe_toggle, filename):
    if contents is not None:
        # Process the image with CLAHE and rotation applied if applicable
        processed_image = parse_contents(contents, brightness, contrast, gamma, clahe_toggle, rotation)

        # Create a Plotly figure with the zoom feature
        fig = px.imshow(processed_image, color_continuous_scale='gray')
        fig.update_layout(
            dragmode="zoom",  # Allows zooming
            margin=dict(l=0, r=0, t=0, b=0),  # Removes default margins
        )

        # Encode the processed image for download
        image_data = encode_image(processed_image)

        # Create a download button
        download_button = html.A(
            dbc.Button("Download Image", color="primary"),
            href=image_data,
            download=f"{filename}" if filename else "downloaded_image.png"
        )

        # Make the graph visible
        return fig, {'display': 'block'}, download_button
    return {}, {'display': 'none'}, ""

# Callback to reset sliders and switch to default values
@app.callback(
    [Output('brightness-slider', 'value'),
     Output('contrast-slider', 'value'),
     Output('gamma-slider', 'value'),
     Output('rotation-slider', 'value'),
     Output('clahe-toggle', 'value')],
    [Input('reset-button', 'n_clicks')]
)
def reset_sliders(n_clicks):
    if n_clicks:
        return 1.0, 1.0, 1.0, 0, False  # Reset all values to defaults, including rotation to 0
    return dash.no_update  # Keep current values if not clicked
