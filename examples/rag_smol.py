import os

import datasets
from smolagents import Tool, CodeAgent, DuckDuckGoSearchTool
from tools.smol import current_time_in_timezone
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
import random
from models.smol import models


class GuestInforRetrieverTool(Tool):
    name = 'guest_info_retriever'
    description = 'Retrieves detailed information about gala guests based on their name or relation.'
    inputs = {
        'query': {
            'type': 'string',
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = 'string'

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return '\n\n'.join([doc.page_content for doc in results[:3]])
        else:
            return 'No matching guest information found.'


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
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
    parquets = 'data/train-*.parquet'
    guest_dataset = datasets.load_dataset('parquet', data_files={'train': parquets}, split='train')

    docs = [
        Document(
            page_content='\n'.join([
                f'Name: {guest["name"]}',
                f'Relations: {guest["relation"]}',
                f'Description: {guest["description"]}',
                f'Email: {guest["email"]}',
            ]),
            metadata={'name': guest['name']}
        )
        for guest in guest_dataset
    ]

    guest_info_tool = GuestInforRetrieverTool(docs)

    alfred = CodeAgent(model=models[model], tools=[current_time_in_timezone, DuckDuckGoSearchTool(), guest_info_tool, WeatherInfoTool()],
                       add_base_tools=True, planning_interval=3)  #, reset=reset)
    response = alfred.run(prompt)
    print(response)
