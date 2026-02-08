# from .PaLM2 import PaLM2
from .GPT import GPT
from .vLLM_local import vLLM_local
from .DeepSeek import DeepSeek
import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()

    if provider == 'gpt':
        model = GPT(config)
    elif provider == 'deepseek':
        model = DeepSeek(config)
    elif provider == 'vllm-local':
        model = vLLM_local(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model