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
        vs = RAG_utils.access_chroma_db(collection_name)

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

    elif opt.vector_store == 'es':
        pass # TODO: implement Elasticsearch retrieval

        return


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

def zephyr_1(question, contexts, opt):
    '''
    Generate answer with Zephyr configuration 1.

    Args:
        question (str): The question to answer.
        contexts (List[str]): The relevant contexts.
        opt (Options): The options for the retrieval.

    Returns:
        str: The answer to the question.
    '''
    zephyr_prompt_template = """<system>{sys_msg}</s>\n<user>{user_msg}</s>\n<|assistant|>"""
    user_msg = opt.user_msg_template.format(context=contexts, question=question)
    prompt_text = zephyr_prompt_template.format(sys_msg=opt.sys_msg, user_msg=user_msg)
    return RAG_utils.gen_text_hf_api(lm_name='HuggingFaceH4/zephyr-7b-beta', prompt_text=prompt_text, temp=0.1, top_k=30, rep_pen=1.03)

def batch_eval(eval_llm, eval_embeddings, opt, batch_size=10, start_from_prev=False):
    '''
    Evaluate a batch of questions with multiple generators.

    RAGAS metrics:
    - Retrieval metrics
        - Context Recall: The extent to which the generated context is relevant to the question (ground_truths, contexts).
            - Not used in this evaluation because it often does not compute.
        - Context Entity Recall: The extent to which the generated context contains relevant entities (ground_truths, contexts).
            - Not used in this evaluation because it often computes only 0 or 1.
        - Context Relevancy: The extent to which the generated context is relevant to the question (question, contexts).
        - Context Precision: The extent to which the generated context is precise to the question (question, ground_truths, contexts).
    - Answer/Context metrics
        - Faithfulness: The extent to which the generated answer is faithful to the context (question, context, answer).
            - Not used in this evaluation because it often computes only 1 or None.
        - Answer Relevancy: The extent to which the generated answer is relevant to the question (question, context, answer).
    - Answer vs. Ground Truth metrics
        - Answer Similarity: The extent to which the generated answer is similar to the ground truth answer (ground truth, answer).
        - Answer Correctness: The extent to which the generated answer is correct (ground truth, answer).

    Args:
        eval_llm (Ollama): The Ollama LLM for evaluation.
        eval_embeddings (OllamaEmbeddings): The Ollama embeddings for evaluation.
        opt (Options): The options for the evaluation.
        batch_size (int): The batch size for evaluation.

    '''
    start_time = time.time()
    qa_df = pd.read_parquet(opt.dataset_path).sample(opt.sample_size, random_state=0)

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

            batch_results = []
            #context_results = []
            with tqdm.tqdm(batch_df.iterrows(), total=batch_df.shape[0], desc='Question', leave=False, file=sys.stdout) as q_bar:
                for j, row in q_bar:
                    question = row['question']
                    ground_truth_answer = row['answer']

                    # retrieve relevant documents
                    contexts = retrieve_contexts(question, opt)
                    context = RAG_utils.format_contexts(contexts)
                    #context_results.append([i, contexts])
                    
                    # Generator loop to not repeat retrieval
                    with tqdm.tqdm(range(n_generators), desc='Generator', leave=False, file=sys.stdout) as gen_bar:
                        for k in gen_bar:
                            # generate answer with LLM
                            answer = opt.generator_funcs[k](question, context, opt).strip()
                            #answer = "asdfasfd"
                            # append results to batch_results
                            func_name = opt.generator_funcs[k].__name__
                            row = [j, func_name, question, answer, contexts, ground_truth_answer]
                            batch_results.append(row)
                
            
            # convert to dataset for evaluation
            batch_results = pd.DataFrame(batch_results, columns=['qa_index','generator','question', 'answer', 'contexts', 'ground_truth'])
            batch_results = Dataset.from_pandas(batch_results)

            # TODO: if multiple generators, first eval retrieval, then loop for generator evals
            # evaluate batch with ragas
            batch_eval_results = ragas.evaluate(batch_results, metrics=[context_relevancy, context_precision, answer_correctness, answer_relevancy, answer_similarity], llm=eval_llm, embeddings=eval_embeddings)

            # append batch results to file
            if not os.path.exists(opt.filepath):
                batch_eval_results.to_pandas().to_csv(opt.filepath, mode='w', header=True, index=False)
            else:
                batch_eval_results.to_pandas().to_csv(opt.filepath, mode='a', header=False, index=False)

            # if i == 2:
            #     break
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
    
    args1 = {'dataset_path': 'rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet', # 'rag-dataset-12000/data/test-00000-of-00001-af2a9f454ad1b8a3.parquet',
            'vector_store': 'chroma',
            'document_embedder': 'all-MiniLM-L6-v2',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'k': 5,
            'generator_funcs': [ollama3_1, zephyr_1],
            'sys_msg': """You are a helpful assistant. Answer the user's question in one sentence based on the provided context. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do NOT start your response with "According to the provided context." """,
            'user_msg_template': """Context: {context} Question: {question}""",
            'sample_size': 10,
            'filepath': 'evals/eval_results.csv',
            }
    opt1 = Options()
    opt1.make_vars(args1)
        
    opt = opt1
    batch_eval(
               eval_llm=langchain_llm, eval_embeddings=langchain_embeddings,
               opt=opt, batch_size=1, start_from_prev=False)