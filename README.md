# CONTROL from Get Smart

[Get Smart Intro on YT](https://www.youtube.com/watch?v=o2ObCoCm61s)

Using local models on ollama and frameworks such as smolagents

#### Create a model with a long context window

    ollama pull gemma3:12b
    ollama create -f modelfile.gemma3

#### Install dependencies

    uv venv
    uv pip install -r requirements.txt