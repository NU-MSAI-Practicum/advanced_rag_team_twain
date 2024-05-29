# advanced_rag_team_twain
## Branches

main : for data science development

deployment-path : for deployment

## Setup
1. Set up environment and packages

`$ python -m venv ragenv`

`$ source ragenv/bin/activate`

`$ pip install -r requirements.txt`

2. Download [dataset](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)

`# Make sure you have git-lfs installed (https://git-lfs.com)`

`$ git lfs install`

`$ git clone https://huggingface.co/datasets/neural-bridge/rag-dataset-12000`

3. Set up Ollama
Download Ollama (see https://ollama.com/)

Download Llama3
`$ ollama pull llama3`

4. While we recommend using Llama3 via Ollama, if you wish to use HuggingFace models, please set your HuggingFace API token.
`os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'hf...'`


5. Build system and evaluate

`python eval.py`
This will chunk the data, build a Chroma vector store (if using Chroma), and evaluate the RAG system. In the main function, predefined default options for Chroma Dense Retrieval, ES Sparse Retrieval, and ES Dense Retrieval are set up.