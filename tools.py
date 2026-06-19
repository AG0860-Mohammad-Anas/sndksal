from langchain_core.tools import tool
from langchain_groq import ChatGroq

from retrievers import (
    load_vector_store,
    get_domain_retriever
)
from config import GROQ_API_KEY, GROQ_MODEL
from prompts import grader_prompt, rewriter_prompt, reflection_prompt
from output_parser import Context_Grading, AnswerReflection

# 1. Initialize Vector Store
try:
    vector_store = load_vector_store()
except Exception:
    vector_store = None

# 2. Initialize LLM specifically for structured output tools
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY
)

# --- BASE RETRIEVAL LOGIC ---
def _search(domain, query):
    if vector_store is None:
        return "Vector database not initialized."

    retriever = get_domain_retriever(vector_store, domain)
    docs = retriever.invoke(query)

    if not docs:
        return "No matching policy found."

    results = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        results.append(
            f"Source File: {source}\nChunk ID: {chunk_id}\n\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)

# --- RETRIEVAL TOOLS ---
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

# --- NEW AGENTIC RAG TOOLS ---
@tool
def grade_retrieved_context(question: str, context: str) -> str:
    """Use this tool immediately after retrieving policy documents to grade if the context is relevant."""
    chain = grader_prompt | llm.with_structured_output(Context_Grading)
    result = chain.invoke({"question": question, "context": context})
    
    if result.decision == "REWRITE_QUERY":
        return f"Context is WEAK. Missing info: {result.missing_information}. ACTION REQUIRED: Use the rewrite_query tool."
    elif result.decision == "ASK_CLARIFICATION":
        return f"Query is ambiguous. ACTION REQUIRED: Ask the user to clarify."
    elif result.decision == "NOT_FOUND":
        return "No relevant policy found. ACTION REQUIRED: Inform the user that the policy does not cover this."
    
    return f"Context is {result.overall_relevance}. Proceed to answer the user's question."

@tool
def rewrite_query(original_question: str, missing_information: str) -> str:
    """Use this tool if the context grader determines the context is WEAK and needs a rewritten query."""
    chain = rewriter_prompt | llm
    result = chain.invoke({"question": original_question, "missing_information": missing_information})
    return f"Rewritten query to search: {result.content}. ACTION REQUIRED: Use the retrieval tools again with this new query."

@tool
def review_answer_grounding(question: str, context: str, proposed_answer: str) -> str:
    """Use this tool BEFORE giving the final answer to the user to check for hallucinations."""
    chain = reflection_prompt | llm.with_structured_output(AnswerReflection)
    result = chain.invoke({"question": question, "context": context, "proposed_answer": proposed_answer})
    
    if result.needs_revision:
        return f"WARNING: Answer needs revision. Unsupported claims: {result.unsupported_claims}. Reflection: {result.reflection_summary}. ACTION REQUIRED: Revise your answer to only use retrieved context."
    return "Answer is fully grounded. Proceed to output the final answer to the user."

# --- REGISTER ALL TOOLS ---
agent_tools = [
    retrieve_hr_policy,
    retrieve_travel_policy,
    retrieve_reimbursement_policy,
    retrieve_it_security_policy,
    retrieve_ai_usage_policy,
    grade_retrieved_context,
    rewrite_query,
    review_answer_grounding
]