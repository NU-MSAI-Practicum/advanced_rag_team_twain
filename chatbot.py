from utils.db.es import search_vector, search_data
from utils import RAG_utils

sparse_index_name = 'rag-dataset-12000-train'
dense_index_name = 'rag-dataset-12000-train-vector'


def get_response(prompt, retrieval_selectbox):
    if retrieval_selectbox == 'Sparse':
        # BM25 Sparse Retreival
        response = search_data(prompt, sparse_index_name, num_results=1)[0]
        res = response['_source']['sentence'] if response else None
    else:
        # Dense Retrieval
        response = search_vector(prompt, dense_index_name, num_results=1)[0]
        res = response['_source']['sentence'] if response else None
        if res:
            # Get generated answer
            system_message = """You are a helpful assistant. Answer the user's question in one sentence based on the provided context. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do NOT start your response with "According to the provided context." """
            user_message_template = """Context: {context} Question: {question}"""
            user_message = user_message_template.format(context=response, question=prompt)

            res = RAG_utils.gen_text_ollama(sys_msg=system_message, user_msg=user_message,
                                               options={'seed': 0, 'temperature': 0.01})
    return res