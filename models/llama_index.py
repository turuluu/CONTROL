from llama_index.llms.ollama import Ollama
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

models = {
    'local': Ollama(model="llama3.2:latest", request_timeout=60.0, additional_kwargs={'num_ctx': 8192}),
    # 'hf': HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct"),
    'hf': None,
    'openai': None
}
