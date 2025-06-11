# TB Detection from Chest X-Ray Images

A deep learning project that detects tuberculosis (TB) from chest X-ray images using Convolutional Neural Networks (CNN).

## 🎥 Live Demo Preview

### Video Demonstration
[![Project Demo](https://img.shields.io/badge/📹_Watch_Demo-Click_Here-blue?style=for-the-badge)](https://github.com/iampiyushchouhan/tb-detection/blob/main/Project%20Deploy%20Preview.webm)

**[📺 View Full Demo Video](https://github.com/iampiyushchouhan/tb-detection/blob/main/Project%20Deploy%20Preview.webm)**

This comprehensive video walkthrough demonstrates:

- **🖼️ Image Upload Process** - How to upload chest X-ray images for analysis
- **🔍 Real-time Analysis** - Live prediction process and model inference
- **📊 Results Display** - How predictions and confidence scores are presented
- **🎯 User Interface** - Complete navigation through the Streamlit application
- **⚡ Performance** - Actual response times and user experience
- **🔄 Multiple Test Cases** - Testing with different X-ray images (Normal vs TB cases)

### Quick Preview
The demo showcases the complete workflow from image upload to TB detection results, highlighting the model's accuracy and the intuitive user interface designed for healthcare professionals and researchers.

> **💡 Tip:** Download the video file to view it locally if GitHub's web player doesn't work optimally in your browser.

---

## 🌐 Live Application
**Demo Link:** [TB Detection System](https://tb-detection-5.onrender.com)

> ⚠️ **First-time load notice:** Due to free tier hosting limitations, the initial request may take up to 60 seconds to respond if the service has been idle. Please be patient during the first load.






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




# **📈 Model Performance**


### Confusion Matrix
<img src="https://github.com/iampiyushchouhan/tb-detection/blob/main/confusion_matrix.png" width="400">

The confusion matrix demonstrates excellent model performance with high accuracy in distinguishing between Normal and TB cases:

- **True Negatives (Normal → Normal):** 698 cases correctly identified as Normal
- **False Positives (Normal → TB):** Only 2 cases incorrectly classified as TB
- **False Negatives (TB → Normal):** 71 cases missed (TB classified as Normal)  
- **True Positives (TB → TB):** 69 cases correctly identified as TB

**Key Metrics:**
- **Accuracy:** ~91.3% overall classification accuracy
- **Precision for TB:** 97.2% (69/71) - Very low false positive rate
- **Recall for TB:** 49.3% (69/140) - Room for improvement in detecting TB cases
- **Specificity:** 99.7% - Excellent at identifying normal cases

### Training Performance
<img src="https://github.com/iampiyushchouhan/tb-detection/blob/main/training_results.png" width="750" height="400">

The training curves reveal important insights about model behavior:

**Accuracy Trends:**
- **Training Accuracy (Blue):** Steadily improves and stabilizes around 95-97%
- **Validation Accuracy (Red):** Shows significant fluctuations with periodic drops
- Clear signs of overfitting as training accuracy remains high while validation accuracy varies dramatically

**Loss Patterns:**
- **Training Loss (Blue):** Consistently decreases and remains low throughout training
- **Validation Loss (Red):** Exhibits high volatility with periodic spikes up to 100
- The erratic validation loss pattern indicates potential overfitting and suggests the need for:
  - Better regularization techniques
  - Data augmentation
  - Early stopping mechanisms
  - Learning rate scheduling


**Model Insights:**
The model shows strong performance on the training set but struggles with generalization, as evidenced by the unstable validation metrics. While the final confusion matrix shows good results, the training curves suggest there's room for improvement in model stability and generalization capability.



# **⚠️ Disclaimer**
This project is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.



# **👨‍💻 Author**

**Piyush Chouhan**  
🔗 GitHub: [@iampiyushchouhan](https://github.com/iampiyushchouhan)  
📧 Feel free to reach out for collaborations or questions!

---

## 🙏 Acknowledgments

We extend our heartfelt gratitude to:

- **Medical Imaging Community** - For providing comprehensive datasets that make research like this possible
- **TensorFlow Team** - For developing robust machine learning frameworks that power our models
- **Streamlit Community** - For creating an intuitive platform that makes ML applications accessible
- **Healthcare Professionals** - Dedicated doctors and researchers working tirelessly on TB detection, treatment, and prevention
- **Open Source Contributors** - All developers who contribute to the tools and libraries that make projects like this feasible

> *"Technology in service of healthcare can save lives. This project is a small step towards making TB detection more accessible and accurate."*

---

# **📄 License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/iampiyushchouhan/tb-detection/issues).

**Made with ❤️ for better healthcare**
