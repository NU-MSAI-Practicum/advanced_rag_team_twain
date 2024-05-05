from utils.db import es_conn
# Run this script carefully!

def truncate_index(index_name):
    # truncate data
    es_conn.indices.delete(index=index_name, ignore=[400, 404])


if __name__ == "__main__":
    index_name = "rag-dataset-12000-train"
    truncate_index(index_name)
