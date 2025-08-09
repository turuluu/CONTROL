"""Throwaway code for figuring out gaia-benchmark"""
from datetime import datetime
import os
import requests
import pandas as pd
from pydantic import BaseModel
from uuid import UUID
from typing import List, Optional
from pathlib import Path
import json

from models.llama_index import models
from tools.llama_index import web_search_tool, wikipedia_tool
from llama_index.core.agent.workflow import AgentWorkflow

from typing import Optional
from pathlib import Path
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from tools.utlz import Tee, Timed

model_key = 'openai'

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")

    def __call__(self, question: str) -> str:
        print(f"[{model_key}] Agent received question (first 50 chars): {question[:50]}...")
        return asyncio.run(run_agent(models[model_key], question))


with open('prompts/gaia_benchmark.txt', 'r', encoding='UTF-8') as prompt_file:
    gaia_prompt = prompt_file.read()


async def run_agent(model, prompt, reset=True):
    agent = AgentWorkflow.from_tools_or_functions(
        [wikipedia_tool],
        llm=model,
        system_prompt=gaia_prompt,
        verbose=True
    )

    response = await agent.run(prompt)
    print(f'{response=}')
    try:
        blocks = response.response.get('blocks')
        if blocks:
            print('-' * 80)
            for block in blocks:
                if 'text' in block:
                    print(block.get('text'))
            print('-' * 80)
    except:
        pass

    return str(response).split('FINAL ANSWER: ')[1]


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


class GaiaTask(BaseModel):
    task_id: UUID
    question: str
    Level: str
    file_name: str
    file_path: Optional[Path] = None


def get_questions() -> List[GaiaTask]:
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    try:
        with open('q.json') as f:
            questions_data = json.load(f)
            print('Fetched questions from "cache"')
    except FileNotFoundError:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            raise Exception("Fetched questions list is empty or invalid format.")
        print(f"Fetched {len(questions_data)} questions.")
        print(f"{type(questions_data)=}")
        print(f"{questions_data=}")

        with open('q.json', 'w') as f:
            json.dump(questions_data, f)

    tasks = [GaiaTask.model_validate(q) for q in questions_data]

    files_url = f"{api_url}/files"
    data_dir = Path('data/tasks')
    data_dir.mkdir(exist_ok=True)
    for t in tasks:
        if t.file_name:
            p = data_dir / t.file_name
            if not p.exists():
                with open(p, 'wb') as f:
                    response = requests.get(f'{files_url}/{t.task_id}', timeout=15)
                    response.raise_for_status()
                    f.write(response.content)
            t.file_path = p

    return tasks


def run_and_submit_all(profile=None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent_smith = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)
    questions_data = get_questions()

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    for item in questions_data:
        task_id = item.task_id
        question_text = item.question
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent_smith(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)
    else:
        print(answers_payload)


if __name__ == "__main__":
    buffer_out = Timed()
    buffer_err = Timed()
    import sys

    tee_out = Tee(buffer_out, sys.stdout)
    tee_err = Tee(buffer_err, sys.stderr)

    with redirect_stdout(tee_out), redirect_stderr(tee_err):
        print()
        print("-" * 30 + " Starting " + "-" * 30)
        run_and_submit_all()

    lines = buffer_out.getvalue().splitlines() + buffer_err.getvalue().splitlines()
    lines.sort()
    log_path = Path.cwd() / 'logs' / 'gaia_output.txt'
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'a+', encoding='UTF-8') as log_file:
        log_file.write('\n'.join(lines))
