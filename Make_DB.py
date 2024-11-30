from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil



def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")


    return chunks


def save_to_chroma(chunks: list[Document]):

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("Initializing HuggingFace model for embeddings...")

    model_name = "sentence-transformers/all-mpnet-base-v2"  
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Create a new DB from the documents
    Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


CHROMA_PATH = "chroma"
DATA_PATH = "temp"
generate_data_store()





