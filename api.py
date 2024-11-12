import torch
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Define the classes
classes = ['0', '1', '2', '3', '4']

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('osteoarthritis_resnet18-model.pth', map_location=device))
model = model.to(device)
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)
        
        # Process and predict
        image = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = classes[predicted_idx.item()]
            class_confidences = {classes[i]: float(probabilities[0][i].item()) * 100 for i in range(len(classes))}

            # Prepare the response
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence.item() * 100,
                'class_confidences': class_confidences
            }
        
        os.remove(image_path)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
