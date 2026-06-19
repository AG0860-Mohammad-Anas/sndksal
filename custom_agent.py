# from typing import TypedDict, List, Optional
# from langgraph.graph import StateGraph, END
# from langchain_groq import ChatGroq
# from langchain_core.documents import Document

# #Importing the configurations, prompts, parsers, and tools
# from config import GROQ_API_KEY, GROQ_MODEL
# from prompts import (
#     classifier_prompt, grader_prompt, rewriter_prompt, 
#     answer_prompt, reflection_prompt, clarification_prompt
# )
# from output_parser import (
#     Query_Classification, Context_Grading, GroundedAnswer, AnswerReflection
# )
# from tools import format_docs_for_prompt, merge_retrieved_contexts
# from retrievers import build_or_load_vector_store, get_domain_retriever

# # Initialize our LLM (Temperature 0 because we want strict, factual answers, no creativity)
# llm = ChatGroq(temperature=0, model_name=GROQ_MODEL, groq_api_key=GROQ_API_KEY)

# # Initialize the Vector Store so our nodes can search it
# # Note: This assumes the vector store has already been built by our ingestion script.
# try:
#     vector_store = build_or_load_vector_store()
# except FileNotFoundError:
#     vector_store = None
#     print("Warning: Vector store not found. Search nodes will fail until ingestion is run.")

# # --- 1. Graph State Definition ---
# # This dictionary tracks everything as the agent moves step-by-step
# class PolicyAgentState(TypedDict):
#     user_question: str
#     query_type: str
#     required_policy_domains: List[str]
#     requires_clarification: bool
#     rewritten_query: Optional[str]
#     retrieved_context: List[Document]
#     context_grade: Optional[dict] # Storing as dict for easy state updates
#     answer_data: Optional[dict]
#     reflection: Optional[dict]
#     retry_count: int
#     final_response: str

# # --- 2. Node Functions ---

# def query_classifier_node(state: PolicyAgentState):
#     print("--- NODE: Query Classifier ---")
#     question = state["user_question"]
    
#     # Force the LLM to output our exact Pydantic schema
#     structured_llm = llm.with_structured_output(Query_Classification)
#     classification = structured_llm.invoke(classifier_prompt.format(question=question))
    
#     print(f"  Classified as: {classification.query_type}")
#     print(f"  Domains needed: {classification.required_policy_domains}")
    
#     return {
#         "query_type": classification.query_type,
#         "required_policy_domains": classification.required_policy_domains,
#         "requires_clarification": classification.requires_clarification
#     }

# def parallel_retrieval_node(state: PolicyAgentState):
#     print("--- NODE: Parallel Retrieval ---")
#     # Use the rewritten query if we had to loop back, otherwise use original
#     query_to_search = state.get("rewritten_query") or state["user_question"]
#     domains = state["required_policy_domains"]
    
#     retrieved_results = {}
    
#     # If the LLM didn't give us specific domains, we search everything
#     if not domains or domains == ["OTHER"]:
#         domains = ["HR_LEAVE", "TRAVEL", "REIMBURSEMENT", "IT_SECURITY", "AI_USAGE"]

#     # Retrieve from multiple domains (FR-11.2: Parallelization Pattern)
#     for domain in domains:
#         print(f"  Searching domain: {domain}")
#         retriever = get_domain_retriever(vector_store, domain)
#         docs = retriever.invoke(query_to_search)
#         retrieved_results[domain] = docs
        
#     merged_docs = merge_retrieved_contexts(retrieved_results)
#     print(f"  Total chunks retrieved: {len(merged_docs)}")
    
#     return {"retrieved_context": merged_docs}

# def context_grader_node(state: PolicyAgentState):
#     print("--- NODE: Context Grader ---")
#     question = state["user_question"]
#     context_str = format_docs_for_prompt(state["retrieved_context"])
    
#     structured_llm = llm.with_structured_output(Context_Grading)
#     grade = structured_llm.invoke(grader_prompt.format(question=question, context=context_str))
    
#     print(f"  Decision: {grade.decision}")
#     return {"context_grade": grade.dict()}

# def query_rewriter_node(state: PolicyAgentState):
#     print("--- NODE: Query Rewriter ---")
#     question = state["user_question"]
#     missing_info = state["context_grade"]["missing_information"]
    
#     rewritten_query = llm.invoke(rewriter_prompt.format(
#         question=question, 
#         missing_information=", ".join(missing_info)
#     )).content
    
#     print(f"  Old query: {question}")
#     print(f"  New query: {rewritten_query}")
    
#     # Increment retry count to prevent infinite loops (FR-8)
#     return {
#         "rewritten_query": rewritten_query,
#         "retry_count": state.get("retry_count", 0) + 1
#     }

# def clarification_response_node(state: PolicyAgentState):
#     print("--- NODE: Clarification Response ---")
#     question = state["user_question"]
#     clarification = llm.invoke(clarification_prompt.format(question=question)).content
#     return {"final_response": clarification}

# def answer_generator_node(state: PolicyAgentState):
#     print("--- NODE: Answer Generator ---")
#     question = state["user_question"]
#     context_str = format_docs_for_prompt(state["retrieved_context"])
    
#     structured_llm = llm.with_structured_output(GroundedAnswer)
#     answer_data = structured_llm.invoke(answer_prompt.format(question=question, context=context_str))
    
#     return {"answer_data": answer_data.dict()}

# def reflection_node(state: PolicyAgentState):
#     print("--- NODE: Reflection / Review ---")
#     question = state["user_question"]
#     context_str = format_docs_for_prompt(state["retrieved_context"])
#     proposed_answer = state["answer_data"]["answer"]
    
#     structured_llm = llm.with_structured_output(AnswerReflection)
#     reflection = structured_llm.invoke(reflection_prompt.format(
#         question=question, context=context_str, proposed_answer=proposed_answer
#     ))
    
#     print(f"  Is Grounded: {reflection.is_grounded}")
#     print(f"  Needs Revision: {reflection.needs_revision}")
#     return {"reflection": reflection.dict()}

# def final_response_node(state: PolicyAgentState):
#     print("--- NODE: Compiling Final Response ---")
#     # If we stopped early due to Not Found or Clarification
#     if "final_response" in state and state["final_response"]:
#          return state

#     # Otherwise, format our structured JSON answer into a readable string
#     ans_data = state["answer_data"]
    
#     # Check if reflection caught an error (simulated revision logic)
#     caveat = ""
#     if state.get("reflection") and state["reflection"]["needs_revision"]:
#         caveat = "\n\n*Note from reviewer: Portions of this answer may lack direct policy evidence.*"

#     sources_text = "\n".join([f"- {s['source_file']} ({s['chunk_id']})" for s in ans_data['sources']])
    
#     final_text = (
#         f"**Answer:** {ans_data['answer']}\n\n"
#         f"**Policy Basis:** {', '.join(ans_data['policy_basis'])}\n\n"
#         f"**Sources:**\n{sources_text}\n\n"
#         f"**Confidence:** {ans_data['confidence']} | **Status:** {ans_data['answerability']}\n\n"
#         f"**Next Step:** {ans_data['recommended_next_step']}"
#         f"{caveat}"
#     )
    
#     return {"final_response": final_text}


# # --- 3. Conditional Routing Functions ---

# def route_after_classifier(state: PolicyAgentState):
#     if state["requires_clarification"]:
#         return "clarification_response_node"
#     return "parallel_retrieval_node"

# def route_after_grader(state: PolicyAgentState):
#     decision = state["context_grade"]["decision"]
#     retries = state.get("retry_count", 0)
    
#     if decision == "ANSWER":
#         return "answer_generator_node"
#     elif decision == "REWRITE_QUERY" and retries < 1: # Max 1 retry
#         return "query_rewriter_node"
#     elif decision == "ASK_CLARIFICATION":
#         return "clarification_response_node"
#     else: # NOT_FOUND or out of retries
#         # If we can't find it, we just pass to answer generator to say "Not found in policy"
#         return "answer_generator_node" 


# # --- 4. Build the Graph ---

# workflow = StateGraph(PolicyAgentState)

# # Add all our nodes
# workflow.add_node("query_classifier_node", query_classifier_node)
# workflow.add_node("parallel_retrieval_node", parallel_retrieval_node)
# workflow.add_node("context_grader_node", context_grader_node)
# workflow.add_node("query_rewriter_node", query_rewriter_node)
# workflow.add_node("clarification_response_node", clarification_response_node)
# workflow.add_node("answer_generator_node", answer_generator_node)
# workflow.add_node("reflection_node", reflection_node)
# workflow.add_node("final_response_node", final_response_node)

# # Add edges (The flow of the program)
# workflow.set_entry_point("query_classifier_node")

# workflow.add_conditional_edges("query_classifier_node", route_after_classifier)
# workflow.add_edge("parallel_retrieval_node", "context_grader_node")
# workflow.add_conditional_edges("context_grader_node", route_after_grader)
# workflow.add_edge("query_rewriter_node", "parallel_retrieval_node")

# workflow.add_edge("answer_generator_node", "reflection_node")
# workflow.add_edge("reflection_node", "final_response_node")
# workflow.add_edge("clarification_response_node", "final_response_node")

# workflow.add_edge("final_response_node", END)

# # Compile the graph into an executable application
# app_graph = workflow.compile()