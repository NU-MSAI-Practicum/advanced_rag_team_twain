# advanced_rag_team_twain

## Setup
1. Set up environment and packages

`$ python -m venv ragenv`

`$ pip install -r requirements.txt`

2. Download [dataset](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)

`# Make sure you have git-lfs installed (https://git-lfs.com)`

`$ git lfs install`

`$ git clone https://huggingface.co/datasets/neural-bridge/rag-dataset-12000`

3. Create documents

`python CreateDocuments.py`
This will chunk the data and save it to a pickle file.

## Testing RAG system
Current tests are in [RAG_testing.ipynb](RAG_testing.ipynb). The functionality is supported by [RAG_utils.py](RAG_utils.py). Initial generator testing done in [RAG_misc_testing.ipynb](RAG_misc_testing.ipynb).
