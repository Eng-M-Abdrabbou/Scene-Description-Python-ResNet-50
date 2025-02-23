from flask import Flask, request, jsonify, render_template
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import os

app = Flask(__name__)

# Download and load Places365 ResNet50 model
def load_places365_model():
    # Create model architecture
    model = models.resnet50(weights=None)
    num_classes = 365
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Download weights if not exists
    weights_path = 'resnet50_places365.pth.tar'
    if not os.path.exists(weights_path):
        url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        response = requests.get(url)
        with open(weights_path, 'wb') as f:
            f.write(response.content)
    
    # Load weights with CPU mapping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# Load model
model = load_places365_model()

# Load Places365 labels
def load_labels():
    labels_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    response = requests.get(labels_url)
    labels = []
    for line in response.text.split('\n'):
        if line:
            label = line.split(' ')[0].split('/')[2:]
            labels.append(' '.join(label))
    return labels

labels = load_labels()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_scene():
    try:
        # Get image from request
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            
        # Get top 5 predictions
        _, pred_idx = torch.topk(output, 5)
        pred_idx = pred_idx[0].tolist()
        
        # Get predicted scene names
        predictions = [
            {
                'scene': labels[idx],
                'confidence': float(output[0][idx])
            }
            for idx in pred_idx
        ]
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
