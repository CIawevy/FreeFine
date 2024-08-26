from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image

class BLIP2():
    def __init__(self, ckpt_path):
        processor = AutoProcessor.from_pretrained(ckpt_path)
        model = Blip2ForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)
        self.processor = processor
        self.model = model
    def __call__(self,image_path, *args, **kwargs):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
