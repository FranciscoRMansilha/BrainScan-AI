from dash import html 
import dash_bootstrap_components as dbc

layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(
                html.H2("Welcome to the BrainScan AI!"),
                className="mb-4"  # Adds margin bottom to separate the header
            )
        ),
        
        # What is BrainScan AI?
        dbc.Row(
            dbc.Col(
                [
                    html.H3("What is BrainScan AI?"),
                    html.P(
                        "BrainScan AI is a proof-of-concept web app demonstrating how artificial intelligence "
                        "can assist in detecting brain tumors from MRI scans. It provides an easy-to-use interface "
                        "for doctors and researchers to explore brain images and make predictions using AI. The app is "
                        "designed to classify four types of brain tumors: Glioma, Meningioma, No Tumor, and Pituitary. "
                        "While the model shows promising results, it is not perfect. There are cases where it may misclassify "
                        "a tumor or miss a specific location, highlighting both the potential and the limitations of AI in the medical field."
                    ),
                ],
                className="mb-5"  # Adds more space after this section
            )
        ),
        
        # How to Use BrainScan AI
        dbc.Row(
            dbc.Col(
                [
                    html.H3("How to Use BrainScan AI"),
                    html.Ol([
                        html.Li([
                            html.B("Download the Dataset: "),
                            "Start by downloading the pre-prepared MRI brain scan images ",
                            html.B(html.A("provided in this link", href="https://drive.google.com/drive/folders/1FPjAxeuXxaeGJw3cmKff2ydxpEQANZV8?usp=sharing", target="_blank")),
                            ". Unzip the folder without altering its structure. These images are part of the test set, "
                            "which the model hasn't seen during training."
                        ]),
                        html.Li([
                            html.B("Explore the Image Analysis Toolkit: "),
                            "Head to the Image Analysis Toolkit page, where you can upload images, adjust brightness, contrast, "
                            "zoom in, and more. Any changes you make can be downloaded directly from the app."
                        ]),
                        html.Li([
                            html.B("Detect Tumors with AI: "),
                            "Visit the Tumor Detection page to upload images for AI analysis. You can view the original image, "
                            "check the predicted tumor class with a bounding box and confidence score, and compare it with the actual labels. "
                            "Additionally, you can download the model's prediction and a LayerCAM heatmap. This heatmap shows which areas of the image"
                            "were most important for the model when making its decision. Areas highlighted in red indicate the parts of the image that the model "
                            "considered most significant in identifying the tumor."
                        ]),
                    ]),
                ],
                className="mb-5"  # Adds space after this section
            )
        ),
        
        # The Model and Dataset
        dbc.Row(
            dbc.Col(
                [
                    html.H3("The Model and Dataset"),
                    html.P(
                        "BrainScan AI uses a YOLOv8 model, fine-tuned to detect brain tumors. The dataset consists of 5249 MRI images, "
                        "labeled with four tumor classes: Glioma, Meningioma, No Tumor, and Pituitary. These images are taken from MRI scans "
                        "in different orientations, angles, and planes (axial, coronal, and sagittal), providing diverse views for model training. "
                        "The model was trained on a portion of these images, while the remaining 40 images in the test set are used for exploration within the app."
                    ),
                    html.P(
                        "The dataset was created by Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, Swati Kanchan, "
                        "and Masoud Nickparvar, and further cleaned by Ahmed Sorour."
                    ),
                    html.P([ 
                        "This dataset is freely available under the ",
                        html.A("CC0: Public Domain license", href="https://creativecommons.org/publicdomain/zero/1.0/", target="_blank"),
                        ". For more information, visit the dataset's creators through ",
                        html.A("Kaggle (Ahmed Sorour)", href="https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes", target="_blank"),
                        ", ",
                        html.A("Kaggle (Masoud Nickparvar)", href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset", target="_blank"),
                        ", and ",
                        html.A("Kaggle (Sartaj, Ankita, Prajakta, Sameer, and Swati)", href="https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri", target="_blank"),
                        "."
                    ]),
                    html.P(
                        html.B("A huge thank you to the dataset creators, as without their work, this app would not have been possible.")
                    ),
                ],
                className="mb-5"  # Adds space after this section
            )
        ),

        # How We Would Improve the App
        dbc.Row(
            dbc.Col(
                [
                    html.H3("How We Would Improve the App"),
                    html.P(
                        "To enhance BrainScan AI, we would actively seek feedback from specialists in the medical field to identify "
                        "additional features that could make the Image Analysis Toolkit more valuable. This may include more advanced "
                        "image processing tools or custom settings that meet real-world needs in radiology."
                    ),
                    html.P(
                        "We also aim to improve the current Explainable AI method by refining LayerCAM and exploring other explainability techniques "
                        "for greater transparency in model decisions. These methods will help doctors better understand the reasoning behind the AI's predictions."
                    ),
                    html.P(
                        "Another priority would be expanding the dataset to improve the model’s accuracy and performance, particularly in handling edge cases "
                        "or misclassifications, if we were successful we would possibly like to extend the dataset to include new tumor classes. Additionally, we would continue to optimize the app's user interface, making it more intuitive and user-friendly."
                    )
                ],
                className="mb-5"  # Adds space after this section
            )
        ),
        
        # Who Made This App
        dbc.Row(
            dbc.Col(
                [
                    html.H3("Who Made This App?"),
                    html.P(
                        "BrainScan AI was developed by three Data Science and AI students: Francisco Mansilha – Concept, dataset preparation, "
                        "model training, and app creation; Lea Banovac – Explainable AI feature development using LayerCAM; and Michał Dziechciarz "
                        "– App deployment and infrastructure."
                    ),
                    html.P(
                        "We'd like to thank the dataset creators for making this project possible. For feedback or inquiries, feel free to reach out to us via LinkedIn:"
                    ),
                    html.Ul([
                        html.Li(html.A("Francisco Mansilha", href="https://www.linkedin.com/in/francisco-mansilha/")),
                        html.Li(html.A("Lea Banovac", href="https://www.linkedin.com/in/lea-banovac-29191a24b/")),
                        html.Li(html.A("Michał Dziechciarz", href="https://www.linkedin.com/in/mdziechciarz/")),
                    ]),
                ],
                className="mb-5"  # Adds space after this section
            )
        ),
    ]
)
