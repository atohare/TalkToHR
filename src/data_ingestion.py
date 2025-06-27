from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredXMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import os
from dotenv import load_dotenv
from .utils.config_loader import get_config

load_dotenv()

def load_documents(temp_filepath):
    """Loads documents from a temporary file path, supporting PDF, DOCX, XML, TXT, and DOC."""
    ext = Path(temp_filepath).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(temp_filepath)
    elif ext == ".docx" or ext == ".doc":
        loader = UnstructuredWordDocumentLoader(temp_filepath)
    elif ext == ".xml":
        loader = UnstructuredXMLLoader(temp_filepath)
    elif ext == ".txt":
        loader = TextLoader(temp_filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def get_embeddings():
    """Initializes and returns the embeddings model."""
    model_config = get_config("model_config")
    embedding_model_info = model_config.get("embedding_model", {})
    model_name = embedding_model_info.get("repo_id", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Specify 'cpu' to avoid trying to use CUDA if not available
    model_kwargs = {'device': 'cpu'} 
    # It's recommended to normalize embeddings for some models
    encode_kwargs = {'normalize_embeddings': False}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def create_vector_store(docs, embeddings, db_path="chroma_db"):
    """Creates a Chroma vector store from documents and embeddings."""
    db = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
    return db

def process_uploaded_files(uploaded_files):
    """
    Processes a list of uploaded PDF files, creating and saving a vector store.
    """
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        # Only save if file does not exist
        if not os.path.exists(temp_filepath):
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        documents = load_documents(temp_filepath)
        docs = split_documents(documents)
        all_docs.extend(docs)

    if all_docs:
        embeddings = get_embeddings()
        create_vector_store(all_docs, embeddings)
        return True
    return False

