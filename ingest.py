import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Get API keys and URL from environment
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DOCS_PATH = "./docs/"
COLLECTION_NAME = "demo_collection"

def ingest_documents():
    print("Starting document ingestion...")
    documents = []
    # Loop through all files in the docs directory
    for filename in os.listdir(DOCS_PATH):
        filepath = os.path.join(DOCS_PATH, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                documents.extend(loader.load())
            elif filename.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(filepath)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    if not documents:
        print("No supported documents found. Exiting.")
        return

    print(f"Loaded {len(documents)} document pages/slides.")

    # The rest of the script is the same
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(all_splits)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("Embedding model loaded.")

    QdrantVectorStore.from_documents(
        documents=all_splits,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True, # Use this to overwrite the collection
    )
    print(f"Successfully added chunks to the '{COLLECTION_NAME}' collection in Qdrant.")

if __name__ == "__main__":
    ingest_documents()