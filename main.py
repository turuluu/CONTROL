import asyncio
from smolagents import LiteLLMModel, HfApiModel
from dotenv import load_dotenv

load_dotenv()

from examples.codeagent_with_search import run as ca_run
from examples.langchain import run as lc_run
from examples.llamaindex_basic import run as li_run
from examples.llamaindex import run as li2_run

hf = HfApiModel()
local = LiteLLMModel(
    max_tokens=1024 * 16,
    temperature=0.5,
    model_id='ollama_chat/128k-gemma3:12b',  # it is possible that this model may be overloaded
    custom_role_conversions=None,
)

# ca_run(hf, 'Who is the president of Burgundy')
# lc_run(hf, "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options.")
# li_run()
asyncio.run(li2_run())

# from examples import multiagents
# from examples import vision