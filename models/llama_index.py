import os
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

models = {
    'local': Ollama(model="qwen3", request_timeout=60.0, additional_kwargs={'num_ctx': 2 * 8192}),
    # 'hf': HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct"),
    'hf': None,
    'openai': OpenAI(
        model="gpt-4o",
        # api_key="something else",  # uses OPENAI_API_KEY env var by default
    )
}
