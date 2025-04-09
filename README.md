# CONTROL from Get Smart

LLM Agents.

For those that didn't start with commodore 64, the names are references to an old TV-parody of secret agents: [Get Smart Intro on YT](https://www.youtube.com/watch?v=o2ObCoCm61s)

Using local models on ollama and frameworks such as smolagents, langchain, etc.

#### Install dependencies

uv

    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt

pip

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### Running on local models using ollama

Install [ollama](https://ollama.com)

#### Create a model with a long context window

1. Create modelfile e.g. modelfile.gemma3
    
        # modelfile for gemma3
        FROM gemma3:12b
        PARAMETER num_ctx 128000
    
2. Create custom variant

        ollama pull gemma3:12b
        ollama create -f modelfile.gemma3 128k-gemma3:12b

Note: The longer the context window, the slower the model gets.

### Running on HuggingFace or OpenAI models

- Create an account on HuggingFace/OpenAI
   - On OpenAI the plus subscription is not enough, you have to prepay a sum to be used for the API calls
- Create a token
- Paste the token to an `.env` file (use the `.env.in` as a template)
