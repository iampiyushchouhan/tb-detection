import streamlit as st
import numpy as np
from PIL import Image

def main():
    st.title("🫁 TB Detection Demo")
    st.write("This is a demo version of the TB detection system")
    
    st.info("📝 **Note**: This is a demonstration. The actual model requires local setup due to size constraints.")
    
    uploaded_file = st.file_uploader("Upload chest X-ray image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray', use_column_width=True)
        
        if st.button('Analyze (Demo)'):
            # Simulate prediction for demo
            demo_result = np.random.choice(['Normal', 'TB Detected'])
            demo_confidence = np.random.uniform(75, 95)
            
            st.subheader("Demo Results:")
            if demo_result == "TB Detected":
                st.error(f"⚠️ **{demo_result}**")
            else:
                st.success(f"✅ **{demo_result}**")
            
            st.write(f"**Confidence**: {demo_confidence:.1f}%")
            st.warning("⚠️ This is a demo with random results. Download and run locally for actual predictions.")
    
    st.markdown("---")
    st.markdown("### 🚀 How to use the full version:")
    st.markdown("""
    1. Clone the repository: `git clone https://github.com/yourusername/tb-detection.git`
    2. Install dependencies: `pip install -r requirements.txt`
    3. Prepare your dataset in the correct folder structure
    4. Train the model: `python tb_detection.py`
    5. Run the app: `streamlit run streamlit_app.py`
    """)

if __name__ == "__main__":
    main()