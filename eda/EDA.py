import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

tqdm.pandas()


RAW_TRAIN_DATA_PATH = 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
RAW_TEST_DATA_PATH = 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'


def EDA():
    
    for raw_data_path in [RAW_TRAIN_DATA_PATH, RAW_TEST_DATA_PATH]:
        if 'train' in raw_data_path:
            dataset = 'Train'
        else:
            dataset = 'Test'

        print('='*30)
        print(f'{dataset} data')
        
        df = pd.read_parquet(raw_data_path)

        for col in df.columns:
            print(f'  {col}')

            # Average Length
            lengths = df[col].str.len()
            print(f'    Average {col} Length: {lengths.mean()}')

            # Max Length
            print(f'    Max {col} Length: {lengths.max()}')

            # Length Distribution
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            sns.histplot(lengths, ax=ax)
            ax.set_title(f'{dataset} {col.capitalize()} Length Distribution')
            ax.set_xlabel(f'{dataset} {col.capitalize()} Length')
            ax.set_ylabel('Frequency')
            plt.savefig('eda/EDAVisualizations/' + dataset + '_' + col.capitalize() + '_length_distribution.png')
            plt.close()

        print('\n\n')


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


def embed_sentence(sentences):
    vectors = []
    for sentence in sentences:
        vector = model.embed_query(sentence)
        vectors.append(vector)
    return vectors


def generate_embeddings(df):
    df['context_sentences'] = df['context'].apply(split_context)
    print ("Finished text splitting.")
    df['context_sentences_vectors'] = df['context_sentences'].progress_apply(embed_sentence)
    df.to_csv("eda/tokenized_docs.csv")

def EDA2():
    generate_embeddings(pd.read_parquet(RAW_TRAIN_DATA_PATH))
    # for raw_data_path in [RAW_TRAIN_DATA_PATH, RAW_TEST_DATA_PATH]:
    #     if 'train' in raw_data_path:
    #         dataset = 'Train'
    #     else:
    #         dataset = 'Test'

    #     print('='*30)
    #     print(f'{dataset} data')
        
    #     df = pd.read_parquet(raw_data_path)

    #     for col in df.columns:
    #         print(f'  {col}')

    #         # Average Length
    #         lengths = df[col].str.len()
    #         print(f'    Average {col} Length: {lengths.mean()}')

    #         # Max Length
    #         print(f'    Max {col} Length: {lengths.max()}')

    #         # Length Distribution
    #         fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #         sns.histplot(lengths, ax=ax)
    #         ax.set_title(f'{dataset} {col.capitalize()} Length Distribution')
    #         ax.set_xlabel(f'{dataset} {col.capitalize()} Length')
    #         ax.set_ylabel('Frequency')
    #         plt.savefig('eda/EDAVisualizations/' + dataset + '_' + col.capitalize() + '_length_distribution.png')
    #         plt.close()

    #     print('\n\n')

# def plot_length():
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     sns.histplot(lengths, ax=ax)
#     ax.set_title(f'{dataset} {col.capitalize()} Length Distribution')
#     ax.set_xlabel(f'{dataset} {col.capitalize()} Length')
#     ax.set_ylabel('Frequency')
#     plt.savefig('eda/EDAVisualizations/' + dataset + '_' + col.capitalize() + '_length_distribution.png')
#     plt.close()



def count_words(text):
    if isinstance(text, str):
        return len(text.split())
    return 0

def EDA3():
    
    for raw_data_path in [RAW_TRAIN_DATA_PATH]:
        # if 'train' in raw_data_path:
        #     dataset = 'Train'
        # else:
        #     dataset = 'Test'

        # print('='*30)
        # print(f'{dataset} data')
        
        df = pd.read_parquet(raw_data_path)

        for col in df.columns:
            print(f'  {col}')

            # Average Length
            lengths = df[col].apply(count_words).values
            print(f'    Average {col} Length: {lengths.mean()}')

            # Max Length
            print(f'    Max {col} Length: {lengths.max()}')

            # Length Distribution
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            sns.histplot(lengths, ax=ax, bins=10)
            ax.set_title(f'{col.capitalize()} Length Distribution', fontsize=20)
            ax.set_xlabel(f'{col.capitalize()} Length', fontsize=15)
            ax.set_ylabel('Frequency', fontsize=15)
            plt.savefig('eda/EDAVisualizations/' + 'Presentation_' + col.capitalize() + '_word_distribution.png')
            plt.close()

        print('\n\n')

if __name__ == '__main__':
    # EDA()
    #EDA2()
    EDA3()