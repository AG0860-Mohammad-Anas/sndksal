from config import EMBEDDING_MODEL,VECTOR_STORE_PATH,TOP_K
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#  callig the embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# building vector_store or load if exist
def build_or_load_vector_store(chunks = None):
    """if get chunks than build the vector db and if not than try to load the existing vector db"""
    embeddings = get_embedding_model()
    
    if chunks:
        print("chunks are provided start building the vector store")
        vector_store = Chroma(
            document = chunks,
            embedding = embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        return vector_store
    else:
        print("no chunks are provided so loading the existing vector store")
        if os.path.exists(VECTOR_STORE_PATH):
            return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        else:
            print("vector store does not exist")
            return FileNotFoundError("Vector store deos not exist")
    
# retriever function
# make cahanges in the retriever for getting ans for the Parallel retrieval pattern

def retriever_domain(vector_store,policy_domain :str):
    """create a specific retriever for sepecific domain"""
    
    retrieve = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "K" : TOP_K,
            "filter" : {"policy_domain" : policy_domain}
        }
    )
    return retrieve