# advanced_rag_team_twain

## Setup
1. Set up environment and packages

`$ python -m venv ragenv`

`$ pip install -r requirements.txt`

2. Download [dataset](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)

`# Make sure you have git-lfs installed (https://git-lfs.com)`

`$ git lfs install`

`$ git clone https://huggingface.co/datasets/neural-bridge/rag-dataset-12000`

3. Set up Ollama
Download Ollama (see https://ollama.com/)

Download Llama3
`$ ollama pull llama3`

4. Build system and evaluate

`python eval.py`
This will chunk the data, build a Chroma vector store, and evaluate the RAG system.
