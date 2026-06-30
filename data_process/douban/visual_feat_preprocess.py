import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

def extract_features(image, model, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the features (the last hidden state)
    features = outputs.last_hidden_state.mean(dim=1).squeeze()
    return features
image = Image.open('11.png')
# Load ViT model and feature extractor
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
features = extract_features(image, model, feature_extractor)
print(features)