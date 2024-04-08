import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_TRAIN_DATA_PATH = 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
RAW_TEST_DATA_PATH = 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'


def join_original_contexts(dataset_path):
    df = pd.read_parquet(dataset_path).head(2) # For testing purposes
    text = df['context'].array
    for i in range(len(text)):
        assert '<sep>' not in text[i]
    text = '<sep>'.join(text)
    return text

def split_into_documents(dataset_path):
    
    original_contexts = join_original_contexts(dataset_path)
    original_contexts = original_contexts.split('<sep>')

    documents = []
    for original_context in original_contexts:
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, # TODO: determine optimal chunk size
            chunk_overlap=20,
        )
        documents.extend(text_splitter.create_documents(original_context))
    return documents
    

if __name__ == '__main__':
    documents = split_into_documents(RAW_TRAIN_DATA_PATH)
    print(len(documents))