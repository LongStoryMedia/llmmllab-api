import os
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
LLAMA_SERVER_EXECUTABLE = os.environ.get(
    "LLAMA_SERVER_EXECUTABLE", "/llama.cpp/build/bin/llama-server"
)
MODELS_FILE_PATH = os.environ.get("MODELS_FILE_PATH", "")
PIPELINE_CACHE_TIMEOUT_MIN = int(os.environ.get("PIPELINE_CACHE_TIMEOUT_MIN", "30"))
PIPELINE_EVICTION_TIMEOUT_MIN = int(
    os.environ.get("PIPELINE_EVICTION_TIMEOUT_MIN", "60")
)
RUNNER_PORT = int(os.environ.get("RUNNER_PORT", "8000"))
RUNNER_HOST = os.environ.get("RUNNER_HOST", "0.0.0.0")
SERVER_PORT_RANGE_START = int(os.environ.get("SERVER_PORT_RANGE_START", "8001"))
SERVER_PORT_RANGE_END = int(os.environ.get("SERVER_PORT_RANGE_END", "8900"))
