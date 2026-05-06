import itertools
import boto3
import chromadb
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction

def get_collection(path, collection_name):
    session = boto3.Session()
    embedding_function = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2:0")

    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name, embedding_function=embedding_function)

    return collection

# 중괄호 { } 제거하고 콜론(:)으로 수정
def get_vector_search_results(collection, question):
    results = collection.query(query_texts=[question], n_results=4)
    return results

# 진입점 함수: 원래 작성하신 경로와 이름 그대로 유지
def get_similarity_search_results(question):
    # 아까 작성하신 "chroma" 폴더와 "my_collection" 이름을 그대로 사용합니다.
    collection = get_collection("chroma", "my_collection")

    search_results = get_vector_search_results(collection, question)

    # 리스트 평탄화 작업
    flattened_results_list = list(itertools.chain(*search_results['documents'])) 
    
    return flattened_results_list
