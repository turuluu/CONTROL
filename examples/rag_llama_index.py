from functools import partial
import datasets
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from models.llama_index import models


async def run(model, prompt, reset=True):
    parquets = 'data/train-*.parquet'
    guests = datasets.load_dataset('parquet', data_files=parquets, split='train')

    docs = [
        Document(
            text='\n'.join([
                f'Name: {guests["name"][i]}',
                f'Relation: {guests["relation"][i]}',
                f'Description: {guests["description"][i]}',
                f'Email: {guests["email"][i]}'
            ]),
            metadata={'name': guests["name"][i]}
        )
        for i in range(len(guests))
    ]

    bm25_retriever = BM25Retriever.from_defaults(nodes=docs)

    def guest_info_retriever(query: str) -> str:
        """Retrieves detailed information about gala guests based on their name or relation."""
        results = bm25_retriever.retrieve(query)
        if results:
            return "\n\n".join([doc.text for doc in results[:3]])
        else:
            return "No matching guest information found."

    guest_info_tool = FunctionTool.from_defaults(guest_info_retriever)

    alfred = AgentWorkflow.from_tools_or_functions(
        [guest_info_tool],
        llm=models[model]
    )

    response = await alfred.run(prompt)
    print(response)
