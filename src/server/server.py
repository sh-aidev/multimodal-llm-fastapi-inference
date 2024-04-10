import uvicorn
import time
from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from threading import Thread
from typing import AsyncGenerator, Dict, Generator, List
import requests
from src.utils.openai_protocol import (ChatCompletionRequest,
                             ChatCompletionResponseStreamChoice,
                             ChatCompletionStreamResponse, DeltaMessage,
                             random_uuid)

from transformers import (TextIteratorStreamer, pipeline)
from src.utils.logger import logger
from src.core.multimodal_llm_inference import LLMMultiModalInference

v1Router = APIRouter()

def get_router(config) -> APIRouter:
    llm = LLMMultiModalInference(model_name=config.llm_config.multi_modal_model.model_name)
    @v1Router.post("/v1/chat/completions")
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
                        image = llm.read_image(url)
                        images.append(image)
                        prompt += f"<image>\\n"
            if message['role'] == "assistant":
                prompt += "ASSISTANT:\\n"
                for content in message['content']:
                    if content['type'] == "text":
                        prompt += f"{content['text']}\\n"

        prompt += "ASSISTANT:\\n"
        
        # print(prompt)

        inputs = llm.processor(text=prompt, images=images if len(images) > 0 else None, return_tensors="pt")
        
        # print(inputs)
        
        inputs['input_ids'] = inputs['input_ids'].to(llm.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(llm.device)
        
        if inputs['pixel_values'] is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(llm.device)

        streamer = TextIteratorStreamer(
            tokenizer=llm.processor,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": True},
        )

        generation_kwargs = dict(
            inputs, max_new_tokens=512, do_sample=True, streamer=streamer
        )

        thread = Thread(target=llm.model.generate, kwargs=generation_kwargs)

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
                    model=config.llm_config.multi_modal_model.model_name,
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
                    model=config.llm_config.multi_modal_model.model_name,
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
                model=config.llm_config.multi_modal_model.model_name,
            )
            data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
            yield f"data: {data}\\n\\n"
            finish_reason_sent[i] = True

            # Send the final done message after all response.n are finished
            yield "data: [DONE]\\n\\n"

        thread.start()

        return StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )

    @v1Router.get("/health")
    def health():
        return {"message": "ok"}
    return v1Router

class APIServer:
    def __init__(self, config, router: APIRouter) -> None:
        self.cfg = config
        self.port = config.llm_config.server.port
        self.host = config.llm_config.server.host

        self.server = FastAPI()
        self.server.add_middleware(
            CORSMiddleware,
            allow_origins = ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            
        )
        self.server.include_router(router, prefix="/v1")
    
    def serve(self):
        uvicorn.run(self.server, port=self.port, host=self.host)