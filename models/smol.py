from smolagents import LiteLLMModel, HfApiModel
local = LiteLLMModel(
        max_tokens=1024 * 16,
        temperature=0.5,
        model_id='ollama_chat/llama3.2:latest',  # it is possible that this model may be overloaded
        custom_role_conversions=None,
    )
models = {
    'local': local,
    'hf': HfApiModel(),
    'openai': None
}
