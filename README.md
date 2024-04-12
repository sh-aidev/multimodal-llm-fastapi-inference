<div align="center">

# Multimodal Large Language Model(LLAVA/BakLLAVA) with FastApi Inference

[![python](https://img.shields.io/badge/-Python_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
![license](https://img.shields.io/badge/License-MIT-green?logo=mit&logoColor=white)

This repository contains the inference code for the multimodal large language model(LLAVA/BakLLAVA) with FastApi backend. The model is based on the [BakLLAVA](https://huggingface.co/llava-hf/bakLlava-v1-hf). FastAPI backend is written in a format such that directly we can call openai api and get the response.
</div>

## 📌 Feature
- [x] Multimodal Large Language Model(LLAVA/BakLLAVA) Inference
- [x] FastApi Backend
- [x] OpenAI like API

## 📁  Project Structure
The directory structure of new project looks like this:
```bash
├── LICENSE
├── README.md
├── __main__.py
├── configs
│   └── config.toml
├── logs
├── requirements.txt
└── src
    ├── __init__.py
    ├── app.py
    ├── core
    │   ├── __init__.py
    │   └── multimodal_llm_inference.py
    ├── pylogger
    │   ├── __init__.py
    │   └── logger.py
    ├── server
    │   ├── __init__.py
    │   ├── client.py
    │   ├── router.py
    │   └── server.py
    └── utils
        ├── __init__.py
        ├── config.py
        ├── logger.py
        ├── models.py
        ├── openai_protocol.py
        └── textformat.py

```

## 🚀 Getting Started

### Step 1: Clone the repository
```bash
git clone https://github.com/sh-aidev/bakLlava-chat-vision.git
cd bakLlava-chat-vision
```

### Step 2: Install the required dependencies
```bash
python3 -m pip install -r requirements.txt
```

### Step 3: Run the FastAPI server
```bash
python3 __main__.py
```

### Step 4: To test the FastAPI server with openai library
```bash
python3 src/server/client.py
```

NOTE: to run the frontend code, please refer to the [frontend](https://github.com/sh-aidev/bakLlava-chat-vision.git) repository.
## 📜  References
- [BakLLAVA](https://huggingface.co/llava-hf/bakLlava-v1-hf)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI](https://beta.openai.com/docs/)
- [Uvicorn](https://www.uvicorn.org/)