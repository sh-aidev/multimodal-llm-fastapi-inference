import os
from src.server.server import APIServer, get_router
from src.utils.logger import logger
from src.utils.config import Config

class App:

    def __init__(self) -> None:
        root_config_dir = "configs"
        logger.debug(f"Root config dir: {root_config_dir}")
        config = Config(root_config_dir)
        router = get_router(config)
        self.server = APIServer(config, router)

    def run(self):
        self.server.serve()