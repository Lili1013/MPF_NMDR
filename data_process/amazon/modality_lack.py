import torch
from PIL import Image
from transformers import DALL_E, DALL_EProcessor, CLIPProcessor, CLIPModel

# Load DALL-E model and processor
dalle_model = DALL_E.from_pretrained("dalle-mini/dalle-mini")
dalle_processor = DALL_EProcessor.from_pretrained("dalle-mini/dalle-mini")

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Function to generate image from text using DALL-E
def generate_image_from_text(text):
    inputs = dalle_processor(text=text, return_tensors="pt")
    with torch.no_grad():
        generated_images = dalle_model.generate(**inputs)
    # Convert tensor to PIL image
    image = Image.fromarray((generated_images[0] * 255).astype("uint8"))
    return image

# Example text prompt
text_description = "A dog is running in a park."

# Generate the image
image = generate_image_from_text(text_description)

# Save or display the generated image
image.save("generated_image.png")
image.show()

# Now let's extract visual features using CLIP
image_inputs = clip_processor(images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    image_features = clip_model.get_image_features(**image_inputs)

# Normalize features
image_features /= image_features.norm(dim=-1, keepdim=True)

# Display the extracted features
print("Visual features shape:", image_features.shape)
