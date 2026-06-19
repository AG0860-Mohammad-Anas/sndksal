import os

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from config import (
    EMBEDDING_MODEL,
    VECTOR_STORE_PATH,
    TOP_K
)


def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def build_vector_store(chunks):

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    return vector_store


def load_vector_store():

    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            "Vector store not found"
        )

    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )


def get_domain_retriever(
    vector_store,
    policy_domain
):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K,
            "filter": {
                "policy_domain": policy_domain
            }
        }
    )