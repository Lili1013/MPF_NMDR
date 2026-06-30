import pandas as pd
import torch

from diffusers import StableDiffusionPipeline

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",torch_dtype= torch.float16)
# pipe = pipe.to("cuda")
#
# prompt = "A cat is running on the park."
# image = pipe(prompt).images[0]
#
# image.save("11.png")
# image.show()

model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
pipe=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16)
pipe = pipe.to("cuda")

to_path = '../datasets/douban_review/book_images/'
df = pd.read_csv('../datasets/douban_review/book/book_multi_modal_data.csv')
for index,row in df.iterrows():
    item_id = row['item_id']
    prompt = row['labels']
    image = pipe(prompt).images[0]
    image.save(to_path+str(item_id)+'.png')

