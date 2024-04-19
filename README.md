# advanced_rag_team_twain


## Setup
0. Set up environment and packages
`$ python -m venv ragenv`

`$ pip install -r requirements.txt`


1. Create documents
`python CreateDocuments.py`
This will chunk the data and save it to a pickle file.

## Testing RAG system
Current tests are in [RAG_Testing.ipynb]. The functionality is supported by [RAG_utils.py]. Initial generator testing done in [RAG_misc_testing.ipynb].
