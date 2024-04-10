import base64
import time
from io import BytesIO
from threading import Thread
from typing import AsyncGenerator, Dict, Generator, List

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          TextIteratorStreamer, pipeline)

from openai_protocol import (ChatCompletionRequest,
                             ChatCompletionResponseStreamChoice,
                             ChatCompletionStreamResponse, DeltaMessage,
                             random_uuid)

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "llava-hf/bakLlava-v1-hf"

print(f"loading model 🧨 {model_name=} ...")

device = torch.device("cuda:0")
dtype = torch.bfloat16

model = LlavaForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device,
)

processor = AutoProcessor.from_pretrained(model_name)

print("model loaded 🚀")

def read_image(input_string):
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

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    chunk_object_type = "chat.completion.chunk"

    request.n = 1  # we will only generate 1 response choice

    prompt = ""
    images = []

    for message in request.messages:
        if message['role'] == "user":
            prompt += "USER:\\n"
            for content in message['content']:
                if content['type'] == "text":
                    prompt += f"{content['text']}\\n"
                if content['type'] == "image_url":
                    # read the image
                    url = content['image_url']['url']
                    image = read_image(url)
                    images.append(image)
                    prompt += f"<image>\\n"
        if message['role'] == "assistant":
            prompt += "ASSISTANT:\\n"
            for content in message['content']:
                if content['type'] == "text":
                    prompt += f"{content['text']}\\n"

    prompt += "ASSISTANT:\\n"
    
    # print(prompt)

    inputs = processor(text=prompt, images=images if len(images) > 0 else None, return_tensors="pt")
    
    # print(inputs)
    
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    
    if inputs['pixel_values'] is not None:
        inputs['pixel_values'] = inputs['pixel_values'].to(device)

    streamer = TextIteratorStreamer(
        tokenizer=processor,
        skip_prompt=True,
        decode_kwargs={"skip_special_tokens": True},
    )

    generation_kwargs = dict(
        inputs, max_new_tokens=512, do_sample=True, streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    def get_role() -> str:
        return "assistant"

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # Send first response for each request.n (index) with the role
        role = get_role()
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name,
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\\n\\n"

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        finish_reason_sent = [False] * request.n

        for res in streamer:
            res: str

            if finish_reason_sent[i]:
                continue

            # Send token-by-token response for each request.n
            delta_text = res
            previous_texts[i] = res
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(content=delta_text), finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name,
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\\n\\n"

        choice_data = ChatCompletionResponseStreamChoice(
            index=i, delta=DeltaMessage(), finish_reason="length"
        )
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[choice_data],
            model=model_name,
        )
        print(chunk)
        data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
        yield f"data: {data}\\n\\n"
        finish_reason_sent[i] = True

        # Send the final done message after all response.n are finished
        yield "data: [DONE]\\n\\n"

    thread.start()

    return StreamingResponse(
        completion_stream_generator(), media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")