from config import CHUNK_OVERLAP, CHUNK_SIZE
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# function for the chunking the documents
def chunking(documents:list[Document]) -> list[Document]:
    """this function take a large documents and split it into smaller chunk"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]  
    )
    
    #calling the recursivetextsplitter

    split_docs = text_splitter.split_documents(documents)
    
    
    # loop for assigning the chunk id
    for i,chunk in enumerate(split_docs):
        base_name = chunk.metadata.get("source_file", "unknown_file")
        chunk.metadata["chunk_id"] = f"{base_name}_chunk_{i}"
        # manually ensure that text is tracked
        chunk.metadata["text"] = chunk.page_content
    print("created chunks from the documents provided")
    return split_docs