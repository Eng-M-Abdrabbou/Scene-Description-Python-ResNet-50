# VisionAid: Scene Detection Web Application

## Project Overview
VisionAid is a web-based scene detection application that uses ResNet-50 pre-trained on the Places365 dataset to identify and classify scenes in real-time using your webcam.

## Features
- Real-time scene detection using webcam
- Top 5 scene predictions with confidence scores
- Simple, user-friendly web interface
- Uses state-of-the-art ResNet-50 pre-trained model

## Technology Stack
- **Backend**: Flask (Python)
- **Machine Learning**: PyTorch, ResNet-50
- **Frontend**: HTML5, JavaScript
- **Model**: Places365 pre-trained weights

## Prerequisites
- Python 3.8+
- pip
- Webcam-enabled device

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Eng-M-Abdrabbou/Scene-Description-Python-ResNet-50.git
cd Scene-Description-Python-ResNet-50
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`

## How It Works
1. The application uses a pre-trained ResNet-50 model from Places365
2. When you click "Detect Scenes", it captures a frame from your webcam
3. The image is processed and sent to the backend for scene classification
4. Top 5 scene predictions are displayed with confidence scores

## Model Details
- **Architecture**: ResNet-50
- **Dataset**: Places365
- **Total Scene Categories**: 365
- **Input Size**: 224x224 pixels
- **Preprocessing**: Resize, Center Crop, Normalize

## Screenshots

<img src="\Images\1.png" width="600" height="350" />

<img src="\Images\2.png" width="600" height="350" />

<img src="\Images\3.png" width="600" height="350" />

## Limitations
- Requires a webcam
- Accuracy depends on lighting and image quality
- Limited to 365 scene categories

## Contributing
Contributions are welcome! Please read the contributing guidelines before getting started.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Places365 Dataset
- ResNet-50 Model
- PyTorch Team
- Flask Framework

## Contact
Mahmoud Abdrabbou - mahmoud.f.abdrabbou@gmail.com
Project Link: https://github.com/Eng-M-Abdrabbou/Scene-Description-Python-ResNet-50.git