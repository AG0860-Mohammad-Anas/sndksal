# Enterprise Policy Assistant with Agentic RAG Using LangGraph

## Project Overview

This project implements an Enterprise Policy Assistant using LangGraph, LangChain, Groq, ChromaDB, and HuggingFace embeddings.

The assistant helps employees retrieve policy-backed answers from enterprise policy documents instead of relying on general LLM knowledge.

Supported policy domains:

* HR Leave Policy
* Travel Policy
* Reimbursement Policy
* IT Security Policy
* AI Usage Policy

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant policy content and generate grounded responses with source references.

---

# Problem Statement

Employees frequently need answers to policy questions such as:

* Can I claim meals during same-day business travel?
* What approvals are required for international travel?
* Can customer data be uploaded to a public AI tool?
* Can I use my personal laptop for office work?

Searching through multiple policy documents manually is time-consuming and often leads to inconsistent interpretations.

The goal of this project is to provide a policy assistant that retrieves relevant policy sections and generates responses based on those documents.

---

# Implementation Choice

### Selected Option

LangGraph Pre-built ReAct Agent

The project follows the Pre-built ReAct Agent option described in the PRD.

Instead of building a fully custom graph workflow, I used LangGraph's prebuilt ReAct agent together with custom retrieval tools for each policy domain.

This approach allowed me to focus on:

* Document ingestion
* Metadata management
* Vector search
* Policy retrieval
* Citation support
* Hallucination reduction

---

# Architecture

```text
User Question
      │
      ▼
LangGraph Prebuilt ReAct Agent
      │
      ▼
Policy Retrieval Tool
      │
      ▼
Chroma Vector Database
      │
      ▼
Relevant Policy Chunks
      │
      ▼
Groq LLM
      │
      ▼
Grounded Response
```

---

# Project Structure

```text
enterprise_policy_agentic_rag/

├── app.py
├── config.py
├── loaders.py
├── chunking.py
├── retrievers.py
├── tools.py
├── prebuilt_agent.py
├── requirements.txt
├── README.md
├── .env

├── policy_data/
│   ├── hr_leave_policy.md
│   ├── travel_policy.md
│   ├── reimbursement_policy.md
│   ├── it_security_policy.md
│   └── ai_usage_policy.md

├── vector_db/

└── outputs/
    ├── sample_run_outputs.md
    └── evaluation_results.json
```

---

# Technology Stack

### Frameworks

* Python
* LangGraph
* LangChain

### LLM

* Groq
* llama-3.3-70b-versatile

### Embeddings

* sentence-transformers/all-MiniLM-L6-v2

### Vector Store

* ChromaDB

### Supporting Libraries

* langchain-community
* langchain-core
* langchain-groq
* langchain-huggingface
* langchain-chroma
* chromadb
* sentence-transformers
* python-dotenv
* pydantic

---

# Document Loading

The application supports:

* .pdf
* .md
* .txt

Policy files are loaded from:

```text
policy_data/
```

Each document is enriched with metadata:

```json
{
  "source_file": "",
  "policy_domain": "",
  "page_number": ""
}
```

This metadata is later used for retrieval and source tracking.

---

# Policy Classification Strategy

Instead of using an LLM classifier during ingestion, a lightweight rule-based classifier was implemented.

The policy dataset follows a predictable naming convention:

```text
hr_leave_policy.md
travel_policy.md
reimbursement_policy.md
it_security_policy.md
ai_usage_policy.md
```

The classification function determines the policy domain directly from the filename.

Example:

```python
if "travel" in filename:
    return "TRAVEL"
```

This approach avoids unnecessary LLM calls and keeps ingestion fast.

---

# Chunking Strategy

Documents are split using RecursiveCharacterTextSplitter.

Configuration:

```text
Chunk Size: 900
Chunk Overlap: 120
```

Each chunk receives a unique identifier:

```text
travel_policy.md_chunk_12
```

Metadata stored per chunk:

```json
{
  "chunk_id": "",
  "source_file": "",
  "policy_domain": ""
}
```

Chunk IDs are used for traceability and citations.

---

# Embeddings and Vector Store

The system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for embedding generation.

All chunks are stored inside ChromaDB.

Benefits:

* Fast similarity search
* Local storage
* Lightweight deployment
* Easy metadata filtering

---

# Retrieval Design

The application implements separate retrieval tools for each policy domain:

* retrieve_hr_policy()
* retrieve_travel_policy()
* retrieve_reimbursement_policy()
* retrieve_it_security_policy()
* retrieve_ai_usage_policy()

Each tool retrieves only documents belonging to its policy domain.

Example:

```python
filter={
    "policy_domain": "TRAVEL"
}
```

This reduces irrelevant retrieval results.

---

# Parallel Retrieval Preparation

Although the implementation uses a prebuilt ReAct agent, metadata and retrieval design were structured to support multi-policy retrieval.

Example question:

```text
Can customer data be uploaded to a public AI tool while working remotely?
```

Potential retrieval sources:

* AI Usage Policy
* IT Security Policy

The policy domain metadata enables retrieval across multiple policy categories when required.

---

# Hallucination Control Strategy

Enterprise policy systems require high factual accuracy.

Several controls were added:

1. Policy documents are the source of truth.
2. The agent is instructed to use retrieval tools before answering.
3. The agent is instructed not to invent policy rules.
4. Missing policy information results in a policy-not-found response.
5. Answers should reference retrieved policy evidence.
6. The system does not guarantee approvals.

Example:

Question:

```text
Will my reimbursement definitely be approved?
```

Expected behavior:

```text
Approval depends on policy conditions,
submitted documentation,
and management/finance review.
```

The assistant should never guarantee approval.

---

# Challenges Faced During Implementation

## 1. Metadata Management

One challenge was ensuring metadata remained available throughout the ingestion and retrieval pipeline.

The PRD requires:

```json
{
  "source_file": "",
  "policy_domain": "",
  "page_number": "",
  "chunk_id": ""
}
```

These fields were manually attached during loading and chunk creation.

---

## 2. PyMuPDF Page Number Handling

While testing PDF ingestion, page metadata was not consistently available in the expected format.

To avoid downstream failures, page metadata handling was added manually and fallback values were introduced where necessary.

This ensured that PDF documents remained compatible with the retrieval pipeline.

---

## 3. Chunk-Level Traceability

The PRD requires chunk-level citations.

To support this, chunk IDs were generated during chunk creation rather than relying on vector store internals.

This made retrieval results easier to inspect and debug.

---

## 4. Retrieval Filtering

Initially retrieval returned unrelated chunks from different policy categories.

Adding policy-domain metadata filtering significantly improved retrieval relevance.

---

# Output Generation

The application automatically generates:

```text
outputs/
├── sample_run_outputs.md
└── evaluation_results.json
```

### sample_run_outputs.md

Stores question-answer pairs from testing sessions.

### evaluation_results.json

Stores structured evaluation records for later review.

---

# Running the Application

## Step 1

Install dependencies:

```bash
uv sync
```

or

```bash
uv pip install -r requirements.txt
```

## Step 2

Create a .env file:

```env
GROQ_API_KEY=your_key

GROQ_MODEL=llama-3.3-70b-versatile

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

POLICY_DATA_PATH=policy_data

VECTOR_STORE_PATH=vector_db

CHUNK_SIZE=900

CHUNK_OVERLAP=120

TOP_K=4
```

## Step 3

Place policy documents inside:

```text
policy_data/
```

## Step 4

Run:

```bash
uv run app.py
```

---

# Sample Questions

1. How many annual leave days can an employee carry forward?
2. Can I claim meals for same-day domestic business travel?
3. What documents are needed for hotel reimbursement?
4. Can I use my personal laptop for office work?
5. What approvals are needed for international travel?
6. Can customer data be uploaded to a public AI tool?
7. Will my reimbursement definitely be approved?
8. What should I do if the policy does not mention my scenario?

---

# Limitations

Current implementation focuses on the Pre-built ReAct Agent architecture.

The following PRD features are not fully implemented:

* Structured query classification
* Context grading node
* Query rewriting
* Reflection node
* Custom LangGraph workflow

These were intentionally excluded to keep the implementation focused on retrieval quality and policy-grounded responses.

---

# Future Improvements

1. Add query classification.
2. Add context grading.
3. Add query rewriting.
4. Add reflection and answer review.
5. Implement custom LangGraph workflow.
6. Add Streamlit UI.
7. Add evaluation dashboard.
8. Support additional enterprise policy domains.

---

# Conclusion

This project demonstrates an Enterprise Policy Assistant built using LangGraph, Groq, ChromaDB, and HuggingFace embeddings.

The system retrieves policy content from local documents and generates grounded responses using a Pre-built ReAct Agent architecture. The design emphasizes retrieval accuracy, metadata traceability, source-backed responses, and reduction of unsupported policy claims.
