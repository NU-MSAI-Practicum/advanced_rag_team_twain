import boto3
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

import sys
import os

# Import yaml_content from config
from config import yaml_content

# Get the absolute path of the project directory
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

assert os.getcwd() == project_path

# Add the project path to the Python path
sys.path.append(project_path)


def get_aws_signature():
    session = boto3.Session()
    credentials = session.get_credentials()
    awsauth = AWS4Auth(credentials.access_key,
                       credentials.secret_key,
                       session.region_name,
                       'es')
    return awsauth


def es_connection(awsauth):
    host = yaml_content['elasticsearch']['practicum']['host']
    port = yaml_content['elasticsearch']['practicum']['port']
    es_conn = Elasticsearch(
        hosts=[{'host': host, 'port': port, 'scheme': 'https'}],
        http_auth=awsauth,
        verify_certs=True,
        node_class='requests',
        timeout=30,
        connection_class=RequestsHttpConnection
    )
    return es_conn


def test_es_connection():
    # Please refer to https://elasticsearch-py.readthedocs.io/en/v7.13.4/ for more functions
    # Build elasticsearch connection
    print(es_conn.info())


awsauth = get_aws_signature()
es_conn = es_connection(awsauth)
