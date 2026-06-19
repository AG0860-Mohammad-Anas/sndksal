from langchain_core.tools import tool
from retrievers import build_or_load_vector_store, retriever_domain

# Load the vector store once so the tools can use it
try:
    vector_store = build_or_load_vector_store()
except FileNotFoundError:
    vector_store = None
    print("Warning: Vector store not found.")

def _fetch_context(domain: str, query: str) -> str:
    """Helper function to run the search and format the results."""
    if not vector_store:
        return "Error: Database not initialized."
    
    retriever = retriever_domain(vector_store, domain)
    docs = retriever.invoke(query)
    
    if not docs:
        return f"No relevant information found."
        
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        formatted.append(f"Source: {source} ({chunk_id})\n: {doc.page_content}")
        
    return "\n\n---\n\n".join(formatted)

# doctring are important for the usifn langchain tool

@tool
def retrieve_hr_policy(query: str) -> str:
    """Use this tool to search the HR policy for rules about leave, vacation, sick days, and carry-forward limits."""
    return _fetch_context("HR_LEAVE", query)

@tool
def retrieve_travel_policy(query: str) -> str:
    """Use this tool to search the Travel policy for rules about domestic/international travel and approvals."""
    return _fetch_context("TRAVEL", query)

@tool
def retrieve_reimbursement_policy(query: str) -> str:
    """Use this tool to search the Reimbursement policy for rules about claiming expenses, meals, and receipts."""
    return _fetch_context("REIMBURSEMENT", query)

@tool
def retrieve_it_security_policy(query: str) -> str:
    """Use this tool to search the IT Security policy for rules about laptops, passwords, and data storage."""
    return _fetch_context("IT_SECURITY", query)

@tool
def retrieve_ai_usage_policy(query: str) -> str:
    """Use this tool to search the AI Usage policy for rules about using public AI tools and customer data."""
    return _fetch_context("AI_USAGE", query)

# list of tools
agent_tools = [
    retrieve_hr_policy,
    retrieve_travel_policy,
    retrieve_reimbursement_policy,
    retrieve_it_security_policy,
    retrieve_ai_usage_policy
]