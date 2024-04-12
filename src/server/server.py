import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.utils.logger import logger
from src.server.router import get_router
from src.utils.config import Config

class APIServer:
    def __init__(self, config: Config) -> None:
        """
        Initialize the API server with the given configuration

        Args:
            config (Config): Configuration object
        """
        logger.debug("Initializing API server")
        self.cfg = config
        self.port = config.llm_config.server.port       # Port to run the server on
        self.host = config.llm_config.server.host       # Host to run the server on

        self.app = FastAPI()
        logger.debug("FastAPI app initialized")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins = ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            
        )
        logger.debug("CORS middleware enabled")

        router = get_router(config)
        logger.debug("Router initialized")

        self.app.include_router(router)
        logger.debug("Router included in the app")
    
    def serve(self):
        logger.debug(f"Server running on {self.host}:{self.port}")
        uvicorn.run(self.app, port=self.port, host=self.host)