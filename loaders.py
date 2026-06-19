import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.documents import Document
from config import POLICY_DATA_PATH

#function for policy categorisation
def policy_classification(filename:str) -> str:
    """A simple function to classified the policies based on the file name """
    filename_lower = filename.lower()
    
    if "hr" in filename_lower or "leave" in filename_lower:
        return "HR_LEAVE"
    elif "travel" in filename_lower:
        return "TRAVEL"
    elif "reimbursement" in filename_lower or "finance" in filename_lower:
        return "REIMBURSEMENT"
    elif "security" in filename_lower or "it" in filename_lower:
        return "IT_SECURITY"
    elif "ai" in filename_lower:
        return "AI_USAGE"
    else:
        return "OTHER"
    
# another function for loading the policies

def load_policies() -> list[Document]:
    """scan the directory path and load the policies"""
    
    all_doc = []
    
    #checking path if exist or not ?
    
    if not os.path.exists(POLICY_DATA_PATH):
        print("Directory path not found")
        return all_doc
    
    #looping through the POLICY_DATA_PATH for loading thr files
    
    for filename in os.listdir(POLICY_DATA_PATH):
        filepath = os.path.join(POLICY_DATA_PATH, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        #now laod the pdf and .md files
        docs = []
        
        if filename.endswith(".md"):
            loader = TextLoader(filepath,encoding="utf-8")
            docs.append(loader.load())
        #now laod the pdf
        elif filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
        #now for the text if provided instead of .md
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding = "utf-8")
            docs = loader.load()
        #skip other
        else:
            print("skipping unsupportyed file")
            
        # metadata
        
        domain = policy_classification(filename)
        
        for doc in docs:
            doc.metadata["source_file"] = filename
            doc.metadata["policy_domain"] = domain
            
            # have to add page number manually casue pymupdf not doing automatically
            if "page"not in doc.metadata:
                doc.metadata["page_number"] = 1
                
            all_doc.append(doc) 
    print("loaded all the docs")
    return all_doc    