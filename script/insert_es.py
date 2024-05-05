import nltk
import time
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from elasticsearch.helpers import bulk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.db import es_conn
from utils.model import embedding_model

tqdm.pandas()
nltk.download('punkt')
CHUNK_SIZE = 1000
OVERLAP_SIZE = 200
EMBEDDING_DIMENSION = 384
SPLIT_TEXT_METHOD = 'chunk'


def split_context_by_sentence(context):
    sentences = sent_tokenize(context)
    return sentences


def split_context_by_chunk(context):
    # use recursive text splitting to split context into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )
    return text_splitter.split_text(context)


def split_context(context):
    if SPLIT_TEXT_METHOD == 'sentence':
        return split_context_by_sentence(context)
    else:
        return split_context_by_chunk(context)


def batch_actions(actions, batch_size):
    total_batches = (len(actions) + batch_size - 1) // batch_size
    for i in range(0, len(actions), batch_size):
        yield i // batch_size, actions[i:i + batch_size], total_batches


def insert_data_bulk(df, index_name):
    df['context_sentences'] = df['context'].apply(split_context)
    print ("finish preprocessing")
    actions = []
    for context_id, context_sentences in enumerate(df['context_sentences']):
        for sentence_id, sentence in enumerate(context_sentences):
            doc_id = f'{context_id}_{sentence_id}'
            body = {
                '_index': index_name,
                '_id': doc_id,
                '_source': {'sentence': sentence}
            }
            actions.append(body)
    batch_size = 500
    for batch_index, batch, total_batches in batch_actions(actions, batch_size):
        print(f"Processing batch {batch_index+1}/{total_batches}")
        bulk(es_conn, batch)
        time.sleep(1)


def embed_sentence(sentences):
    vectors = []
    for sentence in sentences:
        vector = embedding_model.embed_query(sentence)
        vectors.append(vector)
    return vectors


def generate_embeddings(df, index_name):
    df['context_sentences'] = df['context'].apply(split_context)
    print ("Finished text splitting.")
    df['context_sentences_vectors'] = df['context_sentences'].progress_apply(embed_sentence)
    df.to_csv(index_name + "_embeddings")
    print ("Finished embedding.")


def insert_vector_bulk(index_name):
    # load_data
    df = pd.read_csv(index_name + "_embeddings")
    print ("Finished loading data")

    # create the index
    index_body = {
       "settings": {
          "index.knn": True
       },
       "mappings": {
          "properties": {
             "vector": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIMENSION
             },
             "doc_id": {
                "type": "text"
             },
             "sentence": {
                "type": "text"
             },
          }
       }
    }
    if not es_conn.indices.exists(index=index_name):
        es_conn.indices.create(index=index_name, body=index_body)

    # prepare data
    documents = []
    for context_id, row in df.iterrows():
        print(context_id)
        for vector_id, (sentence, vector) in enumerate(zip(eval(row['context_sentences']), eval(row['context_sentences_vectors']))):
            doc_id = f'{context_id}_{vector_id}'
            body = {
                '_index': index_name,
                '_source': {'vector': vector,
                            'doc_id': doc_id,
                            'sentence': sentence}
            }
            documents.append(body)

    # insert in batches
    batch_size = 200  # be careful of request limitation
    for batch_index, batch, total_batches in batch_actions(documents, batch_size):
        print(f"Processing batch {batch_index + 1}/{total_batches}")
        bulk(es_conn, batch)
        time.sleep(3)  # be careful of request limitation


if __name__ == "__main__":
    RAW_TRAIN_DATA_PATH = '../rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
    train_df = pd.read_parquet(RAW_TRAIN_DATA_PATH)

    # # insert texts
    # SPLIT_TEXT_METHOD = 'sentence'
    # index_name = "rag-dataset-12000-train"
    # insert_data_bulk(train_df, index_name)

    # insert vectors
    # index_name = "rag-dataset-12000-train-vector"
    # SPLIT_TEXT_METHOD = 'chunk'  # chunk or sentence
    # generate_embeddings(train_df, index_name)
    # insert_vector_bulk(index_name)
