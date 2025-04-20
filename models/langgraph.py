from langchain_ollama import ChatOllama

models = {
    'local': ChatOllama(model="mistral-nemo", timeout=60, num_ctx=8192),
    # 'hf': HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct"),
    'hf': None,
    'openai': None
}
