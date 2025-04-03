from smolagents import LiteLLMModel
from examples.codeagent_with_search import run

model = LiteLLMModel(
    max_tokens=1024 * 16,
    temperature=0.5,
    model_id='ollama_chat/128k-gemma3:12b',  # it is possible that this model may be overloaded
    custom_role_conversions=None,
)

run(model, 'Who is the president of Burgundy')
