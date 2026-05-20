import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class CNNClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.class_names = ['Normal', 'Loitering', 'Running', 'Intruding', 'Falling']
        self.image_size = 224
        
        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, len(self.class_names))
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, pil_image):
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        return self.class_names[predicted.item()], confidence.item()
