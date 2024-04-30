import nltk
import time
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from elasticsearch.helpers import bulk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.db import es_conn

tqdm.pandas()
nltk.download('punkt')
SPLIT_TEXT_METHOD = 'sentence'  # chunk or sentence
model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def split_context_by_sentence(context):
    sentences = sent_tokenize(context)
    return sentences

def split_context_by_chunk(context):
    # use recursive text splitting to split context into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text([context])


def split_context(context):
    if SPLIT_TEXT_METHOD == 'sentence':
        return split_context_by_sentence(context)
    else:
        return split_context_by_chunk(context)


def batch_actions(actions, batch_size):
    for i in range(0, len(actions), batch_size):
        yield actions[i:i + batch_size]


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
    i = 0
    for batch in batch_actions(actions, batch_size):
        print (i)
        i += 1
        bulk(es_conn, batch)
        time.sleep(1)


def embed_sentence(sentences):
    vectors = []
    for sentence in sentences:
        vector = model.embed_query(sentence)
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
                "dimension": 384
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
    batch_size = 200
    i = 0
    for batch in batch_actions(documents, batch_size):
        print(i)
        bulk(es_conn, batch)
        time.sleep(3)  # be careful of requestion limit
        i += 1


if __name__ == "__main__":
    RAW_TRAIN_DATA_PATH = '../rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
    train_df = pd.read_parquet(RAW_TRAIN_DATA_PATH)

    # # insert texts
    index_name = "rag-dataset-12000-train"
    insert_data_bulk(train_df, index_name)

    # # insert vectors
    # index_name = "rag-dataset-12000-train-vector"
    # generate_embeddings(train_df, index_name)
    # insert_vector_bulk(index_name)
