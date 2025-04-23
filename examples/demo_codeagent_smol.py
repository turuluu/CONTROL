from smolagents import CodeAgent, DuckDuckGoSearchTool, tool
import requests
import yaml
from tools.final_answer import FinalAnswerTool
from tools.web_search import ddg_search
from tools.smol import current_time_in_timezone


# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:  # it's import to specify the return type
    # Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"


final_answer = FinalAnswerTool()


def run(model, prompt):
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)

    agent = CodeAgent(
        model=model,
        tools=[current_time_in_timezone, ddg_search, final_answer],
        ## add your tools here (don't remove final answer)
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates
    )
    agent.run(prompt)
