import os

from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader
)

from langchain_core.documents import Document

from config import POLICY_DATA_PATH


def policy_classification(filename: str) -> str:
    filename = filename.lower()

    if "hr" in filename or "leave" in filename:
        return "HR_LEAVE"

    if "travel" in filename:
        return "TRAVEL"

    if "reimbursement" in filename or "finance" in filename:
        return "REIMBURSEMENT"

    if "security" in filename or "it" in filename:
        return "IT_SECURITY"

    if "ai" in filename:
        return "AI_USAGE"

    return "OTHER"


def load_policies() -> list[Document]:
    docs = []

    if not os.path.exists(POLICY_DATA_PATH):
        return docs

    for filename in os.listdir(POLICY_DATA_PATH):

        filepath = os.path.join(POLICY_DATA_PATH, filename)

        if not os.path.isfile(filepath):
            continue

        loaded_docs = []

        if filename.endswith(".pdf"):
            loaded_docs = PyMuPDFLoader(filepath).load()

        elif filename.endswith(".txt"):
            loaded_docs = TextLoader(
                filepath,
                encoding="utf-8"
            ).load()

        elif filename.endswith(".md"):
            loaded_docs = TextLoader(
                filepath,
                encoding="utf-8"
            ).load()

        else:
            continue

        domain = policy_classification(filename)

        for doc in loaded_docs:
            doc.metadata["source_file"] = filename
            doc.metadata["policy_domain"] = domain

            docs.append(doc)

    return docs