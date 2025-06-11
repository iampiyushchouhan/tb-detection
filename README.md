# TB Detection from Chest X-Ray Images

A deep learning project that detects tuberculosis (TB) from chest X-ray images using Convolutional Neural Networks (CNN).

# ⚠️**Important Note**
**Please Note:** This app is deployed on Render's free tier, which means it goes to sleep after 15 minutes of inactivity. If you're the first visitor after a period of inactivity, please allow 30-60 seconds for the service to wake up. Once active, the application will respond normally.


## 🎯 Project Overview

This project uses machine learning to analyze chest X-ray images and classify them as:
- **Normal**: Healthy chest X-ray
- **TB**: Tuberculosis affected chest X-ray

## 🚀 Features

- **Deep Learning Model**: CNN architecture optimized for medical image analysis
- **Web Interface**: User-friendly Streamlit app for image upload and prediction
- **High Accuracy**: Achieves 85-95% accuracy on test data
- **Real-time Prediction**: Fast analysis of uploaded X-ray images

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Image processing
- **NumPy, Pandas**: Data manipulation
- **Matplotlib, Seaborn**: Data visualization

## 📋 Requirements
- tensorflow>=2.10.0 
- streamlit>=1.25.0 
- numpy>=1.21.0 
- pandas>=1.3.0 
- matplotlib>=3.5.0 
- seaborn>=0.11.0 
- scikit-learn>=1.0.0 
- opencv-python>=4.5.0 
- Pillow>=8.3.0



 ## 📁 Project Files
 
# 🔗 Direct File Links
- [Main Training Script](tb_detection.py) 	- Complete model training code 
- [Streamlit Web App](streamlit_app.py) 	- Production web interface
- [Demo Application](demo_app.py)  			- Demo version for easy deployment
- [Requirements](requirements.txt)  		- Python dependencies
- [Deployment Config](render.yaml)  		- Render.com deployment settings


## 📋 Key Component  

# 🧠 [tb_detection.py](tb_detection.py)  
Main training script that includes:
- CNN model architecture
- Data preprocessing
- Training loop with callbacks
- Model evaluation and visualization

# 🌐 [streamlit_app.py](streamlit_app.py)  
Web interface featuring:
- File upload functionality
- Real-time predictions
- Results visualization
- Model download from Google Drive

# 🎭 [demo_app.py](demo_app.py)  
Demonstration version with:
- Simulated predictions
- No model dependencies
- Easy deployment
- Educational interface

# 🔧 Configuration Files
- [render.yaml](render.yaml) - Render.com deployment configuration
- [.gitignore](.gitignore) - Git ignore patterns
- [LICENSE](LICENSE) - Project license

## 💻 Usage Examples 
# Running the Training Script
 ```bash
# Download and run the main training file
python tb_detection.py
```
**See the complete code in [tb_detection.py](tb_detection.py)**

# Starting the Web Application
```bash
# Run the Streamlit app
streamlit run streamlit_app.py
```
**View the full app code in [streamlit_app.py](streamlit_app.py)**



📈 Model Performance

Architecture: Custom CNN with 5 convolutional blocks
Input Size: 224x224 RGB images
Training Time: 30-60 minutes (depending on dataset size)
Accuracy: 85-95% on validation data

⚠️ Disclaimer
This project is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Piyush Chouhan

GitHub: @iampiyushchouhan


🙏 Acknowledgments

Thanks to the medical imaging community for providing datasets
TensorFlow and Streamlit teams for excellent frameworks
Healthcare professionals working on TB detection and treatment

