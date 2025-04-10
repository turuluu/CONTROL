from llama_index.llms.ollama import Ollama

def run():
    llm = Ollama(model="gemma3:latest", request_timeout=60.0, additional_kwargs={'num_ctx': 100})

    for cpl in llm.stream_complete("What is the capital of France?"):
        print(cpl.delta, end="")