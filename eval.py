from CreateDocuments import create_chunks
import RAG_utils

import sys
import os
import time
import pandas as pd
import tqdm

from datasets import Dataset

import ragas
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, context_recall, context_entity_recall, answer_similarity, context_relevancy, context_precision

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from utils.db.es import search_data, search_vector

def retrieve_contexts(question, opt):
    '''
    Retrieve relevant documents for a given question.

    If the collection does not exist, it is created.

    Args:
        question (str): The question to retrieve relevant documents for.
        opt (Options): The options for the retrieval.

    Returns:
        List[str] of relevant contexts
    '''
    if 'train' in opt.dataset_path:
        dataset = 'train'
    elif 'test' in opt.dataset_path:
        dataset = 'test'

    if opt.vector_store == 'chroma':

        collection_name = dataset + '_' + str(opt.chunk_size) + '_' + str(opt.chunk_overlap) + '_' + opt.document_embedder
        vs = RAG_utils.access_lc_chroma_db(collection_name)

        # if the collection does not exist, create it
        if vs is None:
            print('creating chunks...')
            chunks_file_path = create_chunks(opt.chunk_size, opt.chunk_overlap, opt.dataset_path)
            print('creating vector store...')
            vs = RAG_utils.create_chroma_db(chunks_file_path, collection_name)


        # retrieve relevant documents
        docs = vs.similarity_search_with_relevance_scores(question, k=opt.k)
        contexts = [doc[0].page_content for doc in docs]
        return contexts

    elif opt.vector_store == 'es-sparse':
        # Elasticsearch retrieval BM25
        index_name = 'rag-dataset-12000-train'
        k = 1
        contexts = [context['_source']['sentence'] for context in search_data(question, index_name, k)]
        return contexts
    
    elif opt.vector_store == 'es-dense':
        # Elasticsearch retrieval Dense
        index_name = 'rag-dataset-12000-train-vector'
        contexts = [context['_source']['sentence'] for context in search_vector(question, index_name, opt.k)]
        return contexts


def ollama3_1(question, contexts, opt):
    '''
    Generate answer with Ollama Llama 3 configuration 1.

    Args:
        question (str): The question to answer.
        contexts (List[str]): The relevant contexts.
        opt (Options): The options for the retrieval.

    Returns:
        str: The answer to the question.
    '''
    user_msg = opt.user_msg_template.format(context=contexts, question=question)
    return RAG_utils.gen_text_ollama(sys_msg=opt.sys_msg, user_msg=user_msg, options={'seed':0, 'temperature':0.01})

def batch_eval(eval_llm, eval_embeddings, opt, batch_size=10, start_from_prev=False):
    '''
    Evaluate a batch of questions with multiple generators.

    RAGAS metrics:
    - Retrieval metrics
        - Context Recall: the extent to which the retrieved context aligns with the annotated answer (ground_truths, contexts).
            - Not used in this evaluation because it often does not compute.
        - Context Entity Recall: The extent to which the generated context contains relevant entities (ground_truths, contexts).
            - Not used in this evaluation because it often computes only 0 or 1.
        - Context Relevancy: # of relevant sentences / total # of sentences (question, contexts).
        - Context Precision: If the most relevant contexts are ranked higher (question, ground_truths, contexts).
    - Answer/Context metrics
        - Faithfulness: The extent to which the generated answer is faithful to the context (question, context, answer).
            - Not used in this evaluation because it often computes only 1 or None.
        - Answer Relevancy: assessment of completion and redundancy (question, context, answer).
    - Answer vs. Ground Truth metrics
        - Answer Similarity: Cosine similarity between ground truth and answer (ground truth, answer).
        - Answer Correctness: The extent to which the generated answer is correct (ground truth, answer).

    Args:
        eval_llm (Ollama): The Ollama LLM for evaluation.
        eval_embeddings (OllamaEmbeddings): The Ollama embeddings for evaluation.
        opt (Options): The options for the evaluation.
        batch_size (int): The batch size for evaluation.

    '''
    start_time = time.time()
    qa_df = pd.read_parquet(opt.dataset_path)
    # first `sample_size` rows of qa_df
    qa_df = qa_df.head(opt.sample_size)

    if not start_from_prev:
        if os.path.exists(opt.filepath):
            os.remove(opt.filepath)
        start = 0
    else:
        results_df = pd.read_csv(opt.filepath)
        start = int(results_df.shape[0] / results_df['generator'].nunique())

    n_generators = len(opt.generator_funcs)

    with tqdm.tqdm(range(start, len(qa_df), batch_size), desc='Batch', file=sys.stdout) as batch_bar:
        for i in batch_bar:
            batch_df = qa_df[i:i + batch_size]

            batch_results_basic = []
            batch_results_reranked = []
            with tqdm.tqdm(batch_df.iterrows(), total=batch_df.shape[0], desc='Question', leave=False, file=sys.stdout) as q_bar:
                for j, row in q_bar:
                    question = row['question']
                    ground_truth_answer = row['answer']
                    original_context = row['context']

                    # retrieve relevant documents
                    contexts = retrieve_contexts(question, opt)
                    context = RAG_utils.format_contexts(contexts)

                    # Rerank contexts
                    reranked_contexts = RAG_utils.rerank(question, contexts, threshold=0)
                    reranked_context = RAG_utils.format_contexts(reranked_contexts)
                    n_reranked_contexts = len(reranked_contexts)
                    if n_reranked_contexts == 0:
                        reranked_contexts = ['No relevant contexts.']

                    # Generator loop to not repeat retrieval
                    with tqdm.tqdm(range(n_generators), desc='Generator', leave=False, file=sys.stdout) as gen_bar:
                        for k in gen_bar:
                            # generate answer with LLM
                            answer = opt.generator_funcs[k](question, context, opt).strip()
                            # append results to batch_results
                            func_name = opt.generator_funcs[k].__name__
                            row = [j, opt.vector_store, opt.chunk_size, opt.chunk_overlap, opt.document_embedder, opt.k, func_name, question, original_context, contexts, ground_truth_answer, answer]
                            batch_results_basic.append(row)

                            # generate answer with reranked context
                            reranked_answer = opt.generator_funcs[k](question, reranked_context, opt).strip()
                            row_reranked = [j, opt.vector_store, opt.chunk_size, opt.chunk_overlap, opt.document_embedder, opt.k, n_reranked_contexts, func_name, question, original_context, reranked_contexts, ground_truth_answer, reranked_answer]
                            batch_results_reranked.append(row_reranked)

            # convert to dataset for evaluation
            batch_results_basic = pd.DataFrame(batch_results_basic, columns=['qa_index', 'vector_store', 'chunk_size', 'chunk_overlap', 'doc_embedder', 'k', 'generator', 'question', 'original_context', 'contexts', 'ground_truth', 'answer'])
            batch_results_reranked = pd.DataFrame(batch_results_reranked, columns=['qa_index', 'vector_store', 'chunk_size', 'chunk_overlap', 'doc_embedder', 'k', 'n_reranked_contexts', 'generator', 'question', 'original_context', 'contexts', 'ground_truth', 'answer'])

            # Evaluate basic contexts
            batch_results_basic = Dataset.from_pandas(batch_results_basic)
            batch_eval_results_basic = ragas.evaluate(batch_results_basic, metrics=[context_relevancy, context_precision, answer_correctness, answer_relevancy, answer_similarity], llm=eval_llm, embeddings=eval_embeddings)

            # Evaluate reranked contexts
            batch_results_reranked = Dataset.from_pandas(batch_results_reranked)
            batch_eval_results_reranked = ragas.evaluate(batch_results_reranked, metrics=[context_relevancy, context_precision, answer_correctness, answer_relevancy, answer_similarity], llm=eval_llm, embeddings=eval_embeddings)

            # append batch results to file
            if not os.path.exists(opt.filepath):
                batch_eval_results_basic.to_pandas().to_csv(opt.filepath, mode='w', header=True, index=False)
                batch_eval_results_reranked.to_pandas().to_csv(opt.filepath.replace('.csv', '_reranked.csv'), mode='w', header=True, index=False)
            else:
                batch_eval_results_basic.to_pandas().to_csv(opt.filepath, mode='a', header=False, index=False)
                batch_eval_results_reranked.to_pandas().to_csv(opt.filepath.replace('.csv', '_reranked.csv'), mode='a', header=False, index=False)

    print('=== Evaluation complete ===')
    print('Time elapsed:', time.time() - start_time)


class Options:
    def __init__(self) -> None:
        pass

    def make_vars(self, args: dict):
        for key, val in args.items():
            self.__setattr__(key, val)


if __name__ == '__main__':

    # langchain integrates with ragas
    langchain_llm = Ollama(model="llama3")
    langchain_embeddings = OllamaEmbeddings()

    # # Chroma dense retrieval
    # args1 = {'dataset_path': 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet', # 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet',
    #         'vector_store': 'chroma',
    #         'document_embedder': 'all-MiniLM-L6-v2',
    #         'chunk_size': 1000,
    #         'chunk_overlap': 200,
    #         'k': 5,
    #         'generator_funcs': [ollama3_1],
    #         'sys_msg': """You are a helpful assistant. Answer the user's question in one sentence based on the provided context. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do NOT start your response with "According to the provided context." """,
    #         'user_msg_template': """Context: {context} Question: {question}""",
    #         'sample_size': 1,
    #         'filepath': 'evals/eval_results.csv',
    #         }
    # opt1 = Options()
    # opt1.make_vars(args1)

    # batch_eval(
    #            eval_llm=langchain_llm, eval_embeddings=langchain_embeddings,
    #            opt=opt1, batch_size=1, start_from_prev=False)
    

    # ES sparse retrieval
    args2 = args1 = {'dataset_path': 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet', # 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet',
            'vector_store': 'es-sparse',
            'document_embedder': None,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'k': 1,
            'generator_funcs': [ollama3_1],
            'sys_msg': """You are a helpful assistant. Answer the user's question in one sentence based on the provided context. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do NOT start your response with "According to the provided context." """,
            'user_msg_template': """Context: {context} Question: {question}""",
            'sample_size': 100,
            'filepath': 'evals/sparse_results.csv',
            }
    opt2 = Options()
    opt2.make_vars(args2)


    # args3 = {'dataset_path': 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet', # 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet',
    #         'vector_store': 'es-dense',
    #         'document_embedder': 'all-MiniLM-L6-v2',
    #         'chunk_size': 1000,
    #         'chunk_overlap': 200,
    #         'k': 5,
    #         'rerank': True,
    #         'generator_funcs': [ollama3_1],
    #         'sys_msg': """You are a helpful assistant. Answer the user's question in one sentence based on the provided context. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do NOT start your response with "According to the provided context." """,
    #         'user_msg_template': """Context: {context} Question: {question}""",
    #         'sample_size': 100,
    #         'filepath': 'evals/es_dense_train100.csv',
    #         }
    # opt3 = Options()
    # opt3.make_vars(args3)

    batch_eval(
               eval_llm=langchain_llm, eval_embeddings=langchain_embeddings,
               opt=opt2, batch_size=1, start_from_prev=False)