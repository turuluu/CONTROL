import os
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

class LM(BaseModel):
    mistral_small: str = 'mistral-small3.1'


models = {
    'local': Ollama(model=LM().mistral_small, request_timeout=3 * 60.0, additional_kwargs={'num_ctx': 16 * 1024}),
    # 'hf': HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct"),
    'hf': None,
    'openai': OpenAI(
        model="gpt-4o",
        # api_key="something else",  # uses OPENAI_API_KEY env var by default
    )
}
