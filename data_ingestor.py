from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

embedding_model = OpenAIEmbeddings(openai_api_key=openai_key, model="text-embedding-3-large") 

def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_vector_store(chunks, store_path="Chroma_Indexes"):
    os.makedirs(store_path, exist_ok=True)
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=store_path)
    print(f"✅ Chroma vector store saved to {store_path}")

if __name__ == "__main__":
    print("Loading PDFs...")
    docs = load_documents("data")

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs)

    print(f"{len(chunks)} chunks created. Generating embeddings...")
    create_vector_store(chunks)