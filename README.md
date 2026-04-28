# Multimodal Emotion Recognition System

This project focuses on detecting human emotions using both facial expressions (CSV-based features) and speech signals (audio). It combines machine learning and deep learning models to improve accuracy through multimodal fusion.

The system predicts emotions separately from face and audio, and then combines them to produce a final result.

## Project Overview

- Facial emotion recognition using structured CSV features  
- Speech emotion recognition using MFCC + CNN + BiLSTM  
- Multimodal fusion of predictions  
- Interactive UI using Gradio  

## Dataset Link
Dataset link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-facial-landmark-tracking
Dataset link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## How to Run in Google Colab

### Step 1: Open Notebook
Upload and open the notebook in Google Colab.

### Step 2: Enable GPU
Runtime → Change runtime type → GPU

### Step 3: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Install Required Libraries
```python
!pip install xgboost librosa soundfile tensorflow seaborn gradio joblib
```

### Step 5: Upload Datasets

Face Dataset (CSV files):
/content/drive/MyDrive/Face dataset/

Speech Dataset:
/content/drive/MyDrive/RAVDESS Emotional dataset/

### Step 6: Run Facial Model
Run all cells related to feature extraction, training, and evaluation.

### Step 7: Run Speech Model
Run all cells related to MFCC extraction, CNN+BiLSTM training, and evaluation.

### Step 8: Launch UI
```python
demo.launch(share=True)
```

### Step 9: Use the System
Upload CSV + WAV → Click Predict

Example Output:
FACE EMOTION: fear  
AUDIO EMOTION: happy  
FINAL RESULT: Face: fear | Audio: happy  

## Technologies Used

Python, Scikit-learn, TensorFlow, Librosa, XGBoost, Gradio

## Team Members & Contributions

Nikhilesh Andole (35040343)  
- Speech model pipeline and preprocessing  

Pamulapati Mahesh Babu (35045493)  
- EDA and visualization  

Rajagopala Rao Bandaru (35050526)  
- Feature extraction and dataset creation  

Ashiq Shaik (35033254)  
- CNN + BiLSTM model development  

Mohammed Suleman (35049043)  
- Evaluation, metrics, and model saving  

## Results

- Facial Model Accuracy: ~90%  
- Speech Model Accuracy: ~50–60%  
- Multimodal improves robustness  

## Future Work

- Improve speech model  
- Use transformers  
- Real-time deployment  
