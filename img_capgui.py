#!pip install gradio
import torch
import re
import gradio as gr
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
import warnings
warnings.filterwarnings('ignore')


device='cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)


def predict(image,max_length=64, num_beams=4):
  image = image.convert('RGB')
  image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
  clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
  caption_ids = model.generate(image, max_length = max_length)[0]
  caption_text = clean_text(tokenizer.decode(caption_ids))
  return caption_text


input = gr.Image(label="Upload any Image", type = 'pil')
output = gr.Textbox(label="Captions")
description = "Image Captioning Model"

title = "Deep Learning-based an Automatic Image Captioning System using CV"
interface = gr.Interface(

        fn=predict,
        description=description,
        inputs = input,
        theme="grass",
        outputs=output,
        title=title,
    )
interface.launch(debug=True)

