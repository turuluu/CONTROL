from llama_index.core.agent.workflow import AgentWorkflow
from tools.llama_index import puml_to_png_tool
from models.llama_index import models
from tools.utlz import download_file
from pathlib import Path

def run(model, prompt, reset=True):
    resources = Path.cwd() / 'resources'
    resources.mkdir(exist_ok=True)

    plantuml_docs_url = 'https://pdf.plantuml.net/PlantUML_Language_Reference_Guide_en.pdf'
    docs_path = resources / 'plantuml.pdf'
    if not docs_path.exists():
        download_file(plantuml_docs_url, docs_path)

    if not docs_path.exists():
        print(f'Downloading pdf failed: {plantuml_docs_url=} => {docs_path=}')

    # TODO : rag for pdf
    # TODO : think about diagram
    # TODO : tool for writing puml (codeagent or similar?)
    # TODO : use puml_to_png_tool

