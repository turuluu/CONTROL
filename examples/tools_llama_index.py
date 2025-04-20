from llama_index.core.agent.workflow import AgentWorkflow
from tools.llama_index import ddg_search, weather_info_tool
from models.llama_index import models

async def run(model, prompt, reset=True):
    alfred = AgentWorkflow.from_tools_or_functions(
        [ddg_search, weather_info_tool],
        llm=models[model]
    )

    response = await alfred.run(prompt)
    print(response)