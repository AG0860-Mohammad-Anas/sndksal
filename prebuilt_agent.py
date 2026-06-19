# First run: pip install langgraph
from config import GROQ_API_KEY, GROQ_MODEL
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools import agent_tools

llm = ChatGroq(temperature=0, model_name=GROQ_MODEL, groq_api_key=GROQ_API_KEY)

system_prompt = """You are a strict Enterprise Policy Assistant. Your job is to answer employee questions using ONLY the provided tools.
You MUST search the relevant policies before answering, even if you think you know the answer. 
RULES:  
1. Do not invent policy rules. If it is not in the search results, say you don't know.  
2. ALWAYS cite your sources (File name and chunk ID) in your final answer, no exceptions.  
3. If the user asks a vague question, ask them to clarify first, before you search.  
4. Do not guarantee approvals; say it depends on management/finance review if the policy says so.
"""

# Correct modern construction syntax compiling directly into an executable graph app
app_graph = create_agent(llm, tools=agent_tools, state_modifier=system_prompt)
