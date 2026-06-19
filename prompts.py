from langchain_core.prompts import ChatPromptTemplate

# Query Classifier Prompt
CLASSIFIER_SYSTEM_PROMPT = """You are a router Enterprise Policy Assistant.
Analyze the user query and classify it into the correct policy domains.
Just what policies do we need to search, don't answer the question.
If the query is too vague (e.g. "Can I claim this?"), tag it for clarification

User Question: {question}
"""
classifier_prompt = ChatPromptTemplate.from_template(CLASSIFIER_SYSTEM_PROMPT)

# Context Grader Prompt
GRADER_SYSTEM_PROMPT = """You are a relevance rater.
Compare the user question to the retrieved policy chunks.
Can the context provide an answer to the question?
If it's weak or lacks crucial details, set the decision to REWRITE_QUERY or NOT_FOUND.

User Question: {question}

Retrieved Context:
{context}
"""
grader_prompt = ChatPromptTemplate.from_template(GRADER_SYSTEM_PROMPT)

#Query Rewriter Prompt
REWRITER_SYSTEM_PROMPT = """"Could not find good results in original search query.
Convert the user's question into a highly optimized keyword search string to locate the missing policy information.
Hit key policy terms (ex: instead of “Can I get food money?”, try “meal reimbursement travel limits”)

Original Question: {question}
Missing Information: {missing_information}

Return ONLY the rewritten search string, nothing else.
"""
rewriter_prompt = ChatPromptTemplate.from_template(REWRITER_SYSTEM_PROMPT)

#Answer Generator Prompt
ANSWER_SYSTEM_PROMPT = """You are a strict Enterprise Policy Assistant. Answer the user's question using only the provided retrieved context. 
-Do not invent policy rules, limits, or guarantees. 
-Always cite the source_file and chunk_id. 
-If the policy states it requires review, mention that approval is not guaranteed.

User Question: {question}

Retrieved Context:
{context}
"""
answer_prompt = ChatPromptTemplate.from_template(ANSWER_SYSTEM_PROMPT)

#Reflection Prompt
REFLECTION_SYSTEM_PROMPT = """You are a Compliance Reviewer. Check the proposed answer against the retrieved context, and make sure it lines up with what’s actually in there.
Do not make any guarantees or promises, unless the policy explicitly says so.
Also flag anything that feels ungrounded or hallucinated, meaning claims that were not supported by the retrieved context. If something is only implied or guessed, treat it as suspicious, 
even if it sounds reasonable.

User Question: {question}
Retrieved Context: {context}
Proposed Answer: {proposed_answer}
"""
reflection_prompt = ChatPromptTemplate.from_template(REFLECTION_SYSTEM_PROMPT)

#Clarification Prompt (Simple text output)
CLARIFICATION_SYSTEM_PROMPT = """I can help, but your request is a bit too ambiguous to match against enterprise policy, 
could you tell me the specific details like the travel type 
and the expense type, or anything else that should be considered so I can respond accurately?

User Question: {question}
"""
clarification_prompt = ChatPromptTemplate.from_template(CLARIFICATION_SYSTEM_PROMPT)
