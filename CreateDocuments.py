import os
import numpy as np
import pandas as pd
from pickle import Pickler, Unpickler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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

def split_into_chunks(dataset_path, chunk_size, chunk_overlap):
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
    chunks = text_splitter.create_documents(original_contexts)
   
    return chunks
    
def save_chunks(chunks, dir, dataset_name, chunk_size, chunk_overlap):
    """
    Saves the documents to a pickle file. The filename is based on the dataset, chunk_size, and chunk_overlap.
    
    Args:
        chunks (list): The list of chunks.
        dir (string): The directory where the documents will be saved.
        dataset_name (string): The name of the dataset (train or test).
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        The path to the saved file.
    """

    if not os.path.exists(dir):
        os.makedirs(dir)
    file_path = f'{dir}/{dataset_name}_size_{chunk_size}_overlap_{chunk_overlap}.pkl'
    with open(file_path, 'wb') as file:
        Pickler(file).dump(chunks)
        file.flush()
    print('saved chunks to:', file_path)
    return file_path

def load_chunks(file_path):
    """
    Loads the chunks from a pickle file.

    Args:
        file_path (string): The name of the file.

    Returns:
        The list of chunks.
    """
    with open(file_path, 'rb') as file:
        return Unpickler(file).load()
    
def create_chunks(chunk_size=1000, chunk_overlap=200, dataset_path=RAW_TRAIN_DATA_PATH):
    """
    Creates documents from the dataset and saves them to a pickle file.

    Args:
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        dataset_path (string): The path to the dataset.
    """
    documents = split_into_chunks(dataset_path, chunk_size, chunk_overlap)
    dataset_name = 'train' if 'train' in dataset_path else 'test'
    file_path = save_chunks(documents, 'chunks', dataset_name, chunk_size, chunk_overlap)
    return file_path

def create_embeddings(chunks, model_name):
    """
    Creates embeddings for the chunks using the HuggingFaceEmbeddings from LangChain.

    Args:
        chunks (list): The list of documents.
        model_name (string): The name of the model to use for the embeddings.

    Returns:
        A list of embeddings (list of lists).
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=True)
    doc_func = lambda x: x.page_content
    docs = list(map(doc_func, chunks))
    doc_embeddings = embedding_model.embed_documents(docs) 
    return doc_embeddings

def create_embeddings_from_sentence_transformer(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Creates embeddings for the chunks using the SentenceTransformer from HuggingFace (no LangChain).

    Args:
        chunks (list): The list of chunks.
        model_name (string): The name of the model to use for the embeddings.

    Returns:
        A list of embeddings (tensor).
    """

    model = SentenceTransformer(model_name)
    embeddings = model.encode([doc.page_content for doc in chunks], convert_to_tensor=True)
    return embeddings



if __name__ == '__main__':
    chunk_size = 1000
    chunk_overlap = 100
    for dataset_path in [RAW_TRAIN_DATA_PATH]:
        dataset_name = 'train' if 'train' in dataset_path else 'test'
        chunks = split_into_chunks(dataset_path, chunk_size, chunk_overlap)
        save_chunks(chunks=chunks, dir='chunks', dataset_name=dataset_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # save_chunks(chunks, 'documents', dataset_name, chunk_size, chunk_overlap)
        # embeddings = create_embeddings(chunks, 'all-MiniLM-L6-v2')
        # Alternatively, use generate directly from HF (returns tensors)
        # embeddings2 = create_embeddings_from_sentence_transformer(documents, 'sentence-transformers/all-MiniLM-L6-v2')
