from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def chunking(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):

        source_file = chunk.metadata.get(
            "source_file",
            "unknown"
        )

        chunk.metadata["chunk_id"] = (
            f"{source_file}_chunk_{idx}"
        )

    return chunks