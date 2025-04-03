from smolagents import CodeAgent, DuckDuckGoSearchTool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool


@tool
def web_search(query: str) -> str:
    """A tool that searches the we with the given query
    Args:
        query: the internet search query
    """
    ddg = DuckDuckGoSearchTool()
    try:
        return ddg.forward(query)
    except:
        return "No search results found"

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


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

def run(model, prompt):
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)

    print(len(prompt_templates))

    agent = CodeAgent(
        model=model,
        tools=[get_current_time_in_timezone, web_search, final_answer],  ## add your tools here (don't remove final answer)
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates
    )
    agent.run(prompt)
