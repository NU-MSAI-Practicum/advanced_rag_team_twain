import numpy as np
import pandas as pd
from pickle import Pickler, Unpickler
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_TRAIN_DATA_PATH = 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
RAW_TEST_DATA_PATH = 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'


def get_original_contexts(dataset_path):
    df = pd.read_parquet(dataset_path) # For testing purposes
    print(df.shape)
    df = df.iloc[[7937, 7952, 7771]] # For testing purposes
    return df['context'].array

def split_into_documents(dataset_path, chunk_size, chunk_overlap):
    
    original_contexts = get_original_contexts(dataset_path)

    # https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, # TODO: determine optimal chunk size
            chunk_overlap=chunk_overlap,)
    documents = text_splitter.create_documents(original_contexts)
   
    return documents
    
def save_documents(documents, dir, train, chunk_size, chunk_overlap):
    dataset = 'train' if train else 'test'
    filename = f'{dir}/{dataset}_size_{chunk_size}_overlap_{chunk_overlap}_documents.pkl'
    with open(filename, 'wb') as file:
        Pickler(file).dump(documents)
        file.flush()

def load_documents(filename):
    with open(filename, 'rb') as file:
        return Unpickler(file).load()

if __name__ == '__main__':
    chunk_size = 1000
    chunk_overlap = 200
    #for dataset_path in [RAW_TRAIN_DATA_PATH, RAW_TEST_DATA_PATH]:
    for dataset_path in [RAW_TRAIN_DATA_PATH]:

        documents = split_into_documents(dataset_path, chunk_size, chunk_overlap)
        save_documents(documents, 'documents', 'train' in dataset_path, chunk_size, chunk_overlap)
    
  