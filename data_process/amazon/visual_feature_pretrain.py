import json
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch
from io import BytesIO
import gzip
import pandas as pd
import numpy as np
from loguru import logger


# Load JSON data from file
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


# Download image from URL
def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


# Extract features using ViT model
def extract_features(image, model, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the features (the last hidden state)
    features = outputs.last_hidden_state.mean(dim=1).squeeze()
    return features


# Main function to process images
def process_images(path,to_path):
    # Load JSON data
    # data = load_json(json_path)
    # data =  gzip.open(json_path,'rb')
    data = pd.read_csv(path)

    # Load ViT model and feature extractor
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    image_feature_list = []
    # Iterate over items in the JSON data
    logger.info('start extract features')
    fail_index = []
    num = 0
    for index,item in data.iterrows():
        image_url = item['image_url']
        item_id = item['item_id']

        try:
            # Download and process the image
            image = download_image(image_url)
            features = extract_features(image, model, feature_extractor)
            image_feature_list.append(features.unsqueeze(0))

            # Output or save the features
            # print(f"Item ID: {item_id}, Image Features: {features}")

        except Exception as e:
            fail_index.append(index)
            num += 1
            print(f"Failed to process item {item_id}: {e}")
    image_features = np.vstack(image_feature_list)
    fill_list = [0] * 768
    for each_index in fail_index:
        image_features = np.insert(image_features, each_index, fill_list, axis=0)
    np.save(to_path, image_features)
    logger.info('done!')
    logger.info(f'fail num:{num}')



# Call the main function with your json file path
# path = '../datasets/amazon_review/beauty_multi_modal_data_filter.csv'
# process_images(path,to_path='../datasets/amazon_review/beauty_image_features_filter1.npy')

# path = '../datasets/amazon_review/health_multi_modal_data_filter.csv'
# process_images(path,to_path='../datasets/amazon_review/health_image_features_filter1.npy')

# path = '../datasets/amazon_review/cloth_multi_modal_data_filter.csv'
# process_images(path,to_path='../datasets/amazon_review/cloth_image_features_filter1.npy')

# path = '../datasets/amazon_review/phone_sport_cloth/phone/phone_multi_modal_data_filter.csv'
# process_images(path,to_path='../datasets/amazon_review/phone_sport_cloth/phone/phone_image_features_filter.npy')

# path = '../datasets/amazon_review/phone_sport_cloth/sport/sport_multi_modal_data_filter.csv'
# process_images(path,to_path='../datasets/amazon_review/phone_sport_cloth/sport/sport_image_features_filter.npy')

path = '../datasets/amazon_review/phone_sport_cloth/cloth/cloth_multi_modal_data_filter.csv'
process_images(path,to_path='../datasets/amazon_review/phone_sport_cloth/cloth/cloth_image_features_filter.npy')