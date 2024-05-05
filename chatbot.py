from utils.db.es import search_vector, search_data

sparse_index_name = 'rag-dataset-12000-train'
dense_index_name = 'rag-dataset-12000-train-vector'


def get_response(prompt, retrieval_selectbox):
    if retrieval_selectbox == 'Sparse':
        # BM25 Sparse Retreival
        response = search_data(prompt, sparse_index_name, num_results=1)[0]
    else:
        # Dense Retrieval
        response = search_vector(prompt, dense_index_name, num_results=1)[0]
    return response['_source']['sentence'] if response else None