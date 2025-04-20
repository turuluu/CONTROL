import random
from typing import TypedDict, Annotated
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from models.langgraph import models



def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


def run(model, prompt, reset=True):
    weather_info_tool = Tool(
        name="get_weather_info",
        description="Fetches dummy weather information for a given location.",
        func=get_weather_info,
    )

    tools = [DuckDuckGoSearchRun(), weather_info_tool]
    chat_with_tools = models[model].bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):
        return {
            'messages': [chat_with_tools.invoke(state['messages'])]
        }

    def unwrap_messages(response):
        return response['messages'][-1].content

    builder = StateGraph(AgentState)

    builder.add_node('assistant', assistant)
    builder.add_edge(START, 'assistant')

    builder.add_node('tools', ToolNode(tools))
    builder.add_conditional_edges('assistant', tools_condition)
    builder.add_edge('tools', 'assistant')

    alfred = builder.compile()

    messages = [HumanMessage(content=prompt)]
    response = alfred.invoke({'messages': messages})
    print(unwrap_messages(response))

