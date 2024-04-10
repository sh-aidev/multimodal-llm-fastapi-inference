from pydantic import BaseModel

class LoggerModel(BaseModel):
    environment: str

class ServerModel(BaseModel):
    host: str
    port: int

class MultiModalModel(BaseModel):
    model_name: str

class Model(BaseModel):
    logger: LoggerModel
    server: ServerModel
    multi_modal_model: MultiModalModel