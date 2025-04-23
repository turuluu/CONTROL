from typing import TypedDict, Annotated
import datasets
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from models.langgraph import models

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def run(model, prompt, reset=True):
    parquets = 'data/train-*.parquet'
    guest_dataset = datasets.load_dataset('parquet', data_files={'train': parquets}, split='train')

    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]

    bm25_retriever = BM25Retriever.from_documents(docs)

    def extract_text(query: str) -> str:
        """Retrieves detailed information about gala guests based on their name or relation."""
        results = bm25_retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."

    guest_info_tool = Tool(
        name="guest_info_retriever",
        description="Retrieves detailed information about gala guests based on their name or relation.",
        func=extract_text,
    )

    tools = [guest_info_tool]
    chat_with_tools = models[model].bind_tools(tools)

    def assistant(state: AgentState):
        return {
            'messages': [chat_with_tools.invoke(state['messages'])]
        }

    builder = StateGraph(AgentState)

    builder.add_node('assistant', assistant)
    builder.add_edge(START, 'assistant')

    builder.add_node('tools', ToolNode(tools))
    builder.add_conditional_edges(
        'assistant',
        tools_condition,
    )
    builder.add_edge('tools', 'assistant')

    alfred = builder.compile()

    messages = [HumanMessage(content=prompt)]
    response = alfred.invoke({'messages': messages})
    print (response['messages'][-1].content)
