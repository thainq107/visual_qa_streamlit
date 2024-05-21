import os
import torch
import gdown
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource
def load_model(gdrive_id='11w8Ec-PnvU0Ahw0cOoFTUOvw-ntxdJ33'):

  model_path = 'blip-base'
  if not os.path.exists(model_path):
    # download folder
    gdown.download_folder(id=gdrive_id)
  processor = BlipProcessor.from_pretrained(model_path)
  model = BlipForConditionalGeneration.from_pretrained(model_path)
  return processor, model

processor, model = load_model()

def save_upload_file(upload_file, save_folder='blip-base'):
    os.makedirs(save_folder, exist_ok=True)
    if upload_file:
        new_filename = generate_name()
        save_path = os.path.join(save_folder, new_filename)
        with open(save_path, 'wb+') as f:
            data = upload_file.read()
            f.write(data)

        return save_path
    else:
        raise('Image not found.')

def inference(image, text):
  raw_image = Image.open(image).convert('RGB')
  inputs = processor(raw_image, text, return_tensors="pt")
  out = model.generate(**inputs)
  result = processor.decode(out[0], skip_special_tokens=True)
  return result

def main():
  st.title('Visual Question Answering')
  st.title('Model: BLIP. Dataset: COCO')
  uploaded_img = st.file_uploader('Input Image', type=['jpg', 'jpeg', 'png'])
  example_button = st.button('Run example')
  st.divider()
  
  if example_button:
    uploaded_img_path = 'blip-base/demo.jpg'
  else:
    uploaded_img_path = save_upload_file(uploaded_img)

  prompt = st.text_input("Prompt: ", "A photography of ")
  result = inference(uploaded_img_path, prompt)
  st.image(uploaded_img_path)
  st.success(result) 

if __name__ == '__main__':
     main() 
