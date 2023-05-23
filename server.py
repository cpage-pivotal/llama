import logging
import os
import sys

from flask import Flask, render_template
from llama_index import GPTVectorStoreIndex, GithubRepositoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import RedisVectorStore

app = Flask(__name__)


@app.route('/index')
def hello_world():
    init_env()

    documents = load_data()
    build_index(documents)
    return render_template('index.html')


@app.route('/query')
def query_index():
    init_env()

    index = load_index()
    run_query(index)
    return render_template('index.html')


def init_env():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    try:
        with open('/bindings/openai/key', 'r') as f:
            os.environ["OPENAI_API_KEY"] = f.read()
    except OSError:
        print('No binding for OpenAI Key')

    try:
        with open('/bindings/openai/github_token', 'r') as f:
            os.environ["GITHUB_TOKEN"] = f.read()
    except OSError:
        print('No binding for Github Token')

    try:
        with open('/bindings/redis/host', 'r') as f:
            os.environ["REDIS_HOST"] = f.read()
    except OSError:
        print('No binding for Redis Host')

    try:
        with open('/bindings/redis/port', 'r') as f:
            os.environ["REDIS_PORT"] = f.read()
    except OSError:
        print('No binding for Redis Port')

    try:
        with open('/bindings/redis/password', 'r') as f:
            os.environ["REDIS_PASSWORD"] = f.read()
    except OSError:
        print('No binding for Redis Password')


def define_source_repo():
    github_repo_owner = 'cpage-pivotal'
    github_repo_name = 'ai-data'
    github_branch = 'essay'
    return github_branch, github_repo_name, github_repo_owner


def load_data():
    github_branch, github_repo_name, github_repo_owner = define_source_repo()
    documents = GithubRepositoryReader(
        owner=github_repo_owner,
        repo=github_repo_name,
        use_parser=False,
        verbose=False,
    ).load_data(branch=github_branch)
    return documents


def run_query(index):
    query_engine = index.as_query_engine()
    query_string = "What was the author's relationship to Jessica?"
    print(query_string)
    response = query_engine.query(query_string)
    print(response)


def build_index(documents):
    GPTVectorStoreIndex.from_documents(documents, storage_context=get_storage_context())


def load_index():
    index = GPTVectorStoreIndex([], storage_context=get_storage_context())
    return index


def get_storage_context():
    index_prefix = "llama"
    index_name = "pg_essays"
    redis_url = get_redis_url()
    print("Connecting to " + redis_url)

    vector_store = RedisVectorStore(
        index_name=index_name,
        index_prefix=index_prefix,
        redis_url=redis_url,
        overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context


def get_redis_url():
    if os.getenv("REDIS_PASSWORD") is None:
        prefix = "redis://"
    else:
        prefix = "redis://default:" + os.getenv("REDIS_PASSWORD") + "@"
    return prefix + os.getenv("REDIS_HOST") + ":" + os.getenv("REDIS_PORT")


if __name__ == '__main__':
    app.run()
