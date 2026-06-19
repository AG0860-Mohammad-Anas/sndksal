from langchain_core.tools import tool

from retrievers import (
    load_vector_store,
    get_domain_retriever
)

try:
    vector_store = load_vector_store()
except Exception:
    vector_store = None


def _search(domain, query):

    if vector_store is None:
        return "Vector database not initialized."

    retriever = get_domain_retriever(
        vector_store,
        domain
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No matching policy found."

    results = []

    for doc in docs:

        source = doc.metadata.get(
            "source_file",
            "unknown"
        )

        chunk_id = doc.metadata.get(
            "chunk_id",
            "unknown"
        )

        results.append(
            f"""
Source File: {source}
Chunk ID: {chunk_id}

{doc.page_content}
"""
        )

    return "\n\n---\n\n".join(results)


@tool
def retrieve_hr_policy(query: str):
    """Search HR leave policies."""
    return _search("HR_LEAVE", query)


@tool
def retrieve_travel_policy(query: str):
    """Search travel policies."""
    return _search("TRAVEL", query)


@tool
def retrieve_reimbursement_policy(query: str):
    """Search reimbursement policies."""
    return _search("REIMBURSEMENT", query)


@tool
def retrieve_it_security_policy(query: str):
    """Search IT security policies."""
    return _search("IT_SECURITY", query)


@tool
def retrieve_ai_usage_policy(query: str):
    """Search AI usage policies."""
    return _search("AI_USAGE", query)


agent_tools = [
    retrieve_hr_policy,
    retrieve_travel_policy,
    retrieve_reimbursement_policy,
    retrieve_it_security_policy,
    retrieve_ai_usage_policy
]