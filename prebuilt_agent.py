from langchain.agents import create_agent

from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL
)

from tools import agent_tools


llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY
)

SYSTEM_PROMPT = """
You are a strict Enterprise Policy Assistant.

Rules:

1. Always use tools before answering.
2. Never invent policy rules.
3. If policy information is unavailable, say so.
4. Always cite source file names and chunk IDs.
5. Ask clarification questions when needed.
6. Never guarantee approvals.
"""

app_graph = create_agent(
    model=llm,
    tools=agent_tools,
    system_prompt=SYSTEM_PROMPT
)