import time
from streamlit.runtime.scriptrunner import get_script_run_ctx
from utils import RAG_utils
from utils.db.es import search_vector, search_data, es_conn

sparse_index_name = 'rag-dataset-12000-train'
dense_index_name = 'rag-dataset-12000-train-vector'


def get_session_id():
    session_id = get_script_run_ctx().session_id
    return session_id


def get_response(prompt, retrieval_selectbox):
    start = time.time()
    if retrieval_selectbox == 'Sparse':
        # BM25 Sparse Retreival
        response = search_data(prompt, sparse_index_name, num_results=1)[0]
        res = response['_source']['sentence'] if response else None
        retrieval_end = time.time()
        generator_time = 0
    else:
        # Dense Retrieval
        response = search_vector(prompt, dense_index_name, num_results=5)
        retrieval_end = time.time()
        if response:
            response = ("\n\n").join([res['_source']['sentence'] for res in response])
            system_message = """You are a helpful assistant. Answer the user's question in one sentence based on the provided context. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Do NOT start your response with "According to the provided context." """
            user_message_template = """Context: {context} Question: {question}"""
            user_message = user_message_template.format(context=response, question=prompt)
            res = RAG_utils.gen_text_ollama(sys_msg=system_message, user_msg=user_message,
                                               options={'seed': 0, 'temperature': 0.01})
            generator_end = time.time()
            generator_time = generator_end - retrieval_end
    # Logging
    retrieval_time = retrieval_end - start
    total_time = retrieval_time + generator_time
    log_entry = {
        "session_id": get_session_id(),
        "user_input": prompt,
        "answer": res,
        "retrieval_method": retrieval_selectbox,
        "retrieval_time": retrieval_time,
        "generator_time": generator_time,
        "total_time": total_time,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    es_conn.index(index="logs", body=log_entry)
    return res