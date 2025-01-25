import numpy as np
import cv2
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Load the skin type model
skin_type_model = load_model("skin_type_model.keras")

# Severity levels and their mapped integer values
severity_mapping = {"Clear": 1, "Mild": 2, "Moderate": 3, "Severe": 4, "Very Severe": 5}

# Preprocess uploaded image for skin type prediction
def preprocess_image(image, img_size=(128, 128)):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR format
    img = cv2.resize(img, img_size)  # Resize to match dataset image size
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to predict skin type directly
def predict_skin_type_directly(image):
    img = preprocess_image(image)
    img = img.reshape(1, 128, 128, 3)  # Reshape for the model
    prediction = skin_type_model.predict(img)
    class_labels = ['dry', 'normal', 'oily']
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class.upper()  # Convert to uppercase for display

# Dummy function for acne severity prediction (replace with actual model or logic)
def predict_acne_severity(image):
    # Placeholder: Return a random severity level
    severity_levels = ["Clear", "Mild", "Moderate", "Severe", "Very Severe"]
    return np.random.choice(severity_levels)

# Function to navigate between pages
def go_to_page(page_name):
    st.session_state["page"] = page_name

# Define CSS style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #3C3C3C;
    }
    h1, h2, h3, h4, h5, h6, p, div, label {
        color: #FFFFFF; 
        font-size: 20px;  
    }
    .stTitle h1 {
        font-size: 60px !important;  
        font-weight: bold;  
        color: #FFFFFF;  
    }
    .stButton button {
        background-color: #00cc96;  
        color: white;
        border-radius: 15px;
        border: none;
        padding: 15px 30px;  
        font-size: 18px;  
        font-weight: bold;
    }
    .stFileUploader {
        color: #000000;  
        border-radius: 10px;  
        font-size: 16px;  
    }
    .stFileUploader .uploaded-file-text {
        color: #000000;  
    }
    .stButton button:hover {
        background-color: #00b386;  
    }
    
    .block-container .col {
        margin-right: 50px;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Home page
def show_home_page():
    st.title("Regular Skin Care Routine")  
    st.write("A guide to regular skin care with product recommendations and routines.")

    col1, col2 = st.columns([0.7, 0.5])  

    with col1:
        image = Image.open("wallpaper.jpg")  
        st.image(image, caption='Regular Skin Care', use_container_width=True)

    with col2:
        st.write(""" 
        Maintaining a consistent skin care routine can make a huge difference in your skin's health.
        Make sure to stick to your routine daily and choose the right products for your skin type.
        """)
        if st.button("Get Started"):
            go_to_page("upload_page")  

# Upload page with skin type and severity prediction
def show_upload_page():
    st.title("Upload or Take a Picture")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Image uploaded successfully!")

        # Predict skin type directly
        predicted_skin_type = predict_skin_type_directly(image)

        # Display predicted skin type
        st.markdown(f"<h3 style='color: white; font-size: 28px;'>Predicted Skin Type:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: blue; font-weight: bold; text-align: center;'>{predicted_skin_type}</h2>", unsafe_allow_html=True)

        # Predict acne severity using a placeholder (replace with actual logic/model)
        predicted_severity_label = predict_acne_severity(image)

        # Display predicted acne severity
        st.markdown(f"<h3 style='color: white; font-size: 28px;'>Predicted Acne Severity:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: red; font-weight: bold; text-align: center;'>{predicted_severity_label}</h2>", unsafe_allow_html=True)

        # Use severity mapping to get the integer value for the slider
        severity_int_value = severity_mapping.get(predicted_severity_label, 1)
        st.slider("Predicted Acne Severity Level", min_value=1, max_value=5, value=severity_int_value, step=1, disabled=True)
        
    if st.button("Go Back"):
        go_to_page("home_page")

# Page navigation logic
if "page" not in st.session_state:
    st.session_state["page"] = "home_page"

if st.session_state["page"] == "home_page":
    show_home_page()
elif st.session_state["page"] == "upload_page":
    show_upload_page()
