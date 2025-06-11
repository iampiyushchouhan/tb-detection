import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
@st.cache_resource
def load_tb_model():
    try:
        model = load_model('tb_detection_final_model.h5')
        return model
    except:
        return None

def predict_tb(image, model):
    # Preprocess image
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        result = "TB Detected"
        confidence = prediction * 100
    else:
        result = "Normal"
        confidence = (1 - prediction) * 100
    
    return result, confidence

# Main app
def main():
    st.title("🫁 TB Detection from Chest X-Ray")
    st.write("Upload a chest X-ray image to detect tuberculosis")
    
    # Load model
    model = load_tb_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first by running: python tb_detection.py")
        return
    else:
        st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray', use_column_width=True)
        
        # Make prediction
        if st.button('🔍 Analyze X-ray'):
            with st.spinner('Analyzing...'):
                result, confidence = predict_tb(image, model)
                
                # Display results
                st.subheader("📊 Analysis Results:")
                
                if result == "TB Detected":
                    st.error(f"⚠️ **{result}**")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    st.write("🏥 **Recommendation:** Please consult with a healthcare professional immediately.")
                else:
                    st.success(f"✅ **{result}**")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    st.write("ℹ️ **Note:** This appears to be a normal chest X-ray.")
                
                # Disclaimer
                st.warning("⚠️ **Important:** This tool is for educational purposes only and should not replace professional medical diagnosis.")

if __name__ == "__main__":
    main()