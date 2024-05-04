import os
import numpy as np
import pandas as pd
from pickle import Pickler, Unpickler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


RAW_TRAIN_DATA_PATH = 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
RAW_TEST_DATA_PATH = 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'


def get_original_contexts(dataset_path):
    """
    Loads the dataset and returns the 'context' column as a numpy array.
    The context column contains the context on which the query/responses are based.

    Args:
        dataset_path: The path to the dataset.

    Returns:
        An array containing the 'context' column.
    """

    df = pd.read_parquet(dataset_path)
    #print(df.shape)
    #df = df.iloc[[7937, 7952, 7771]] # For testing purposes
    return df['context'].array

def split_into_documents(dataset_path, chunk_size, chunk_overlap):
    """
    Splits the contexts into documents. It uses the RecursiveCharacterTextSplitter to split the contexts into chunks.
    The RecursiveCharacterTextSplitter strategy maximizes the length of each document.
    See: https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/

    Args:
        dataset_path (string): The path to the dataset.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        A list of documents, where each document is a list of chunks.
    """
    
    original_contexts = get_original_contexts(dataset_path)

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,)
    documents = text_splitter.create_documents(original_contexts)
   
    return documents
    
def save_documents(documents, dir, train, chunk_size, chunk_overlap):
    """
    Saves the documents to a pickle file. The filename is based on the dataset, chunk_size, and chunk_overlap.
    
    Args:
        documents (list): The list of documents.
        dir (string): The directory where the documents will be saved.
        train (bool): True if the documents are from the training set, False otherwise.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
    """

    dataset = 'train' if train else 'test'
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = f'{dir}/{dataset}_size_{chunk_size}_overlap_{chunk_overlap}_documents.pkl'
    with open(filename, 'wb') as file:
        Pickler(file).dump(documents)
        file.flush()

def load_documents(filename):
    """
    Loads the documents from a pickle file.

    Args:
        filename (string): The name of the file.

    Returns:
        The list of documents.
    """
    with open(filename, 'rb') as file:
        return Unpickler(file).load()
    

def create_embeddings(documents, model_name):
    """
    Creates embeddings for the documents using the HuggingFaceEmbeddings from LangChain.

    Args:
        documents (list): The list of documents.
        model_name (string): The name of the model to use for the embeddings.

    Returns:
        A list of embeddings (list of lists).
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=True)
    doc_func = lambda x: x.page_content
    docs = list(map(doc_func, documents))
    doc_embeddings = embedding_model.embed_documents(docs) 
    return doc_embeddings


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


from sentence_transformers import SentenceTransformer

def create_embeddings_from_sentence_transformer(documents, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Creates embeddings for the documents using the SentenceTransformer from HuggingFace (no LangChain).

    Args:
        documents (list): The list of documents.
        model_name (string): The name of the model to use for the embeddings.

    Returns:
        A list of embeddings (tensor).
    """

    model = SentenceTransformer(model_name)
    embeddings = model.encode([doc.page_content for doc in documents], convert_to_tensor=True)
    return embeddings


if __name__ == '__main__':
    chunk_size = 1000
    chunk_overlap = 200
    #for dataset_path in [RAW_TRAIN_DATA_PATH, RAW_TEST_DATA_PATH]:
    for dataset_path in [RAW_TRAIN_DATA_PATH]:
        documents = split_into_documents(dataset_path, chunk_size, chunk_overlap)
        #save_documents(documents, 'documents', 'train' in dataset_path, chunk_size, chunk_overlap)
        embeddings = create_embeddings(documents, 'all-MiniLM-L6-v2')
        # Alternatively, use generate directly from HF (returns tensors)
        # embeddings2 = create_embeddings_from_sentence_transformer(documents, 'sentence-transformers/all-MiniLM-L6-v2')
