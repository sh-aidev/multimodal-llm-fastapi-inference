import torch
from src.utils.logger import logger
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          TextIteratorStreamer, pipeline)
import base64
from io import BytesIO
from PIL import Image
import requests

class LLMMultiModalInference:
    def __init__(self, 
                 model_name: str = "llava-hf/bakLlava-v1-hf",
                 ) -> None:
        logger.debug(f"Loading model 🧨 {model_name}")
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.model = model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True, device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.debug(f"Model {model_name} loaded successfully 🚀")

    def read_image(input_string: str) -> Image:
        if input_string.startswith("http"):
            # Case: URL
            response = requests.get(input_string)
            img = Image.open(BytesIO(response.content))
        elif input_string.startswith("data:image"):
            # Case: base64-encoded string
            _, encoded_data = input_string.split(",", 1)
            img_data = base64.b64decode(encoded_data)
            img = Image.open(BytesIO(img_data))
        else:
            raise ValueError("Unsupported input format")

        return img
    
    def run(self):
        return self.model