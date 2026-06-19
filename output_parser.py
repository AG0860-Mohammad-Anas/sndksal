from typing import List, Literal
from pydantic import BaseModel, Field

class Query_Classification(BaseModel):
    query_type : str = Field(description="One of: HR_LEAVE, TRAVEL, REIMBURSEMENT, IT_SECURITY, AI_USAGE, MULTI_POLICY, UNANSWERABLE, AMBIGUOUS, OTHER")
    required_policy_domains: List[str] = Field(description="A List of required domains, for example :- ['HR_LEAVE', 'TRAVEL']")
    requires_parallel_retrieval: bool = Field(description="True ,if multiple policy domains are needed")
    requires_clarification: bool = Field(description="True, if the user's question is too vague to search")
    reasoning_summary: str = Field(description="In short explanation of why this classification was chosen")

class Context_Grading(BaseModel):
    overall_relevance: Literal["HIGHLY_RELEVANT", "PARTIALLY_RELEVANT", "WEAK", "NOT_RELEVANT"]
    relevant_chunks: List[str] = Field(description="List of chunk_ids that contain useful information")
    irrelevant_chunks: List[str] = Field(description="List of chunk_ids that do not help")
    missing_information: List[str] = Field(description="List of concepts missing from the context needed to answer")
    decision: Literal["ANSWER", "REWRITE_QUERY", "ASK_CLARIFICATION", "NOT_FOUND"]

class Source_Citation(BaseModel):
    source_file: str = Field(description="The name of the file where information was found")
    policy_domain: str
    chunk_id: str
    snippet: str

class GroundedAnswer(BaseModel):
    answer: str = Field(description="The direct answer to the user's question.")
    policy_basis: List[str] = Field(description="Key policy rules used to form the answer.")
    sources: List[Source_Citation] = Field(description="List of citations linking to the exact retrieved chunks.")
    answerability: Literal["ANSWERED", "PARTIALLY_ANSWERED", "NOT_FOUND", "NEEDS_CLARIFICATION"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    recommended_next_step: str = Field(description="What the user should do next, e.g., 'Submit a claim'.")


class AnswerReflection(BaseModel):
    is_grounded: bool = Field(description="True if the answer is completely supported by the retrieved text.")
    has_citations: bool = Field(description="True if sources are cited properly.")
    unsupported_claims: List[str] = Field(description="List of any claims made that aren't in the context.")
    needs_revision: bool = Field(description="True if the answer hallucinates or needs fixing.")
    reflection_summary: str = Field(description="Brief explanation of the review.")
    