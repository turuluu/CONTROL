import asyncio
from dotenv import load_dotenv

load_dotenv()

model_types = ['local', 'hf', 'openai']

from examples.demo_codeagent_smol import run as ca_run
from examples.demo_simple_langchain import run as lc_run
from examples.demo_simple_llama_index import run as li2_run
from examples.demo_tools_llama_index import run as li_tools_example
from examples.demo_rag_smol import run as rag_run
from examples.demo_rag_llama_index import run as li_rag_example
from examples.demo_rag_langgraph import run as lg_rag_example
from examples.demo_tools_langgraph import run as lg_tools_example

# ca_run(hf, 'Who is the president of Burgundy')
# lc_run(hf, "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options.")
# li_run()
# asyncio.run(li2_run())
rag_prompt = 'Tell me about our guest named "Lady Ada Lovelace".'
# rag_run('local', rag_prompt)
# asyncio.run(li_rag_example('local', rag_prompt))
# lg_rag_example('local',rag_prompt)

tools_prompt = 'What\'s the weather like in Paris tonight? Will it be suitable for our fireworks display?'
# asyncio.run(li_tools_example('local', tools_prompt))
# At the time of writing this, duckduck_search is called with backend='auto' by the langchain_community wrapper
# which does not resolve the backend correctly, unless you have lxml installed
# quick alternative for a fix
#     change the line 31 of .venv/lib/python3.12/site-packages/langchain_community/utilities/duckduckgo_search.py
#     from => to
#     backend: str = "auto" => backend: str = "api"
# lg_tools_example('local', tools_prompt)

# from examples import multiagents
# from examples import vision

from tools.llama_index import puml_to_png

puml_to_png('', '')