import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('./final_model.h5')

def preprocess_image(image):
    # Resize the image to 252x252
    image = image.resize((252, 252))
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    # Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=[0, -1])
    return img_array

def postprocess_mask(mask):
    # Convert probabilities to binary mask
    return (mask > 0.5).astype(np.uint8) * 255

st.title('Liver and Tumor Segmentation')

uploaded_file = st.file_uploader("Choose a CT image...", type="png")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded CT Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Postprocess the prediction
    liver_mask = postprocess_mask(prediction[0, :, :, 0])
    tumor_mask = postprocess_mask(prediction[0, :, :, 1])

    # Display the results
    col1, col2 = st.columns(2)
    with col1:
        st.image(liver_mask, caption='Liver Segmentation', use_column_width=True)
    with col2:
        st.image(tumor_mask, caption='Tumor Segmentation', use_column_width=True)

    # Optionally, you can provide download buttons for the masks
    liver_img = Image.fromarray(liver_mask)
    tumor_img = Image.fromarray(tumor_mask)

    buffer_liver = io.BytesIO()
    liver_img.save(buffer_liver, format="PNG")
    buffer_tumor = io.BytesIO()
    tumor_img.save(buffer_tumor, format="PNG")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Liver Segmentation",
            data=buffer_liver.getvalue(),
            file_name="liver_segmentation.png",
            mime="image/png"
        )
    with col2:
        st.download_button(
            label="Download Tumor Segmentation",
            data=buffer_tumor.getvalue(),
            file_name="tumor_segmentation.png",
            mime="image/png"
        )