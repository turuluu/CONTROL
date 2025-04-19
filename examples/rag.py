import os

import datasets
from smolagents import Tool, CodeAgent
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from pathlib import Path

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

def run(model, prompt):
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

    guest_info_tool =GuestInforRetrieverTool(docs)

    alfred = CodeAgent(model=model, tools=[guest_info_tool])
    response = alfred.run(prompt)
    print(response)

