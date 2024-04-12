import torch
from src.utils.logger import logger
from transformers import (AutoProcessor, LlavaForConditionalGeneration)
import base64
from io import BytesIO
from PIL import Image
import requests

class LLMMultiModalInference:
    """
    Class to handle multimodal inference using a LLM model. This class will handle the model loading and inference.

    """
    def __init__(self, 
                 model_name: str = "llava-hf/bakLlava-v1-hf",
                 ) -> None:
        """
        Initialize the multimodal inference class with the given model name

        Args:
            model_name (str, optional): Model name to load. Defaults to "llava-hf/bakLlava-v1-hf".
        """

        logger.debug(f"Loading model 🧨 {model_name}")
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.model_name = model_name
        self.model = model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True, device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.debug(f"Model {model_name} loaded successfully 🚀")

    def read_image(self, input_string: str) -> Image:
        """
        Read an image from the given input string. The input string can be a URL or a base64-encoded string.

        Args:
            input_string (str): Input string to read the image from

        Returns:
            Image: PIL Image object
        """
        if input_string.startswith("http"):
            # Case: URL
            response = requests.get(input_string)
            logger.debug(f"Image response received: {response}")
            img = Image.open(BytesIO(response.content))
            logger.debug(f"Image opened successfully")
        elif input_string.startswith("data:image"):
            # Case: base64-encoded string
            _, encoded_data = input_string.split(",", 1)
            img_data = base64.b64decode(encoded_data)
            logger.debug(f"Image data decoded")
            img = Image.open(BytesIO(img_data))
            logger.debug(f"Image opened successfully")
        else:
            logger.error(f"Unsupported input format: {input_string}")
            raise ValueError("Unsupported input format")

        return img
