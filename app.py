import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain import hub

# Corrected Qdrant Imports for the new library version
from qdrant_client import QdrantClient, models

# Load environment variables
load_dotenv()

# --- SETUP ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# --- SESSION STATE INITIALIZATION ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'processed_file_names' not in st.session_state:
    st.session_state.processed_file_names = []
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None

# --- CORE CACHED FUNCTIONS ---
@st.cache_resource
def get_models():
    """Initializes and returns the LLM and the local embedding model."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=GOOGLE_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

# In your app.py, replace the entire function with this one.

def ingest_and_create_retriever(uploaded_files, session_id):
    if not uploaded_files: return None

    docs = []
    temp_dir = f"temp_files_{session_id}"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
        loader_map = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".pptx": UnstructuredPowerPointLoader}
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext in loader_map:
            loader = loader_map[file_ext](temp_path)
            docs.extend(loader.load())

    if not docs: return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    _, embeddings = get_models()
    texts = [doc.page_content for doc in splits]
    metadatas = [dict(doc.metadata, tenant=session_id) for doc in splits]

    # ---> STEP 1: Create the collection and add documents.
    # This LangChain method will automatically create the collection if it doesn't exist.
    # This is the crucial first step.
    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=False
    )

    # ---> STEP 2: Create the payload index.
    # Now that we are sure the collection exists, we can create the index.
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.tenant",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    # ---> STEP 3: Get the vector store and create the retriever.
    # Now we can safely connect to the existing, indexed collection.
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    
    return vector_store.as_retriever(
        search_kwargs={'filter': models.Filter(must=[
            models.FieldCondition(key="metadata.tenant", match=models.MatchValue(value=session_id))
        ])}
    )

# --- STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬")
st.title("AI Research Assistant")

with st.sidebar:
    st.header("Your Documents")
    uploaded_files = st.file_uploader(
        "Upload files for this session:", accept_multiple_files=True, type=['pdf', 'docx', 'pptx']
    )
    st.info(f"Session ID: {st.session_state.session_id}")

# --- AGENT AND TOOL LOGIC ---
current_file_names = sorted([f.name for f in uploaded_files])
if uploaded_files and current_file_names != st.session_state.processed_file_names:
    with st.spinner("Processing documents... This may take a minute the first time."):
        st.session_state.retriever = ingest_and_create_retriever(uploaded_files, st.session_state.session_id)
        st.session_state.processed_file_names = current_file_names
        st.session_state.agent_executor = None
    st.success("Documents processed successfully!")

if not st.session_state.agent_executor:
    tools = [DuckDuckGoSearchRun(name="web_search")]
    if st.session_state.retriever:
        document_tool = create_retriever_tool(
            st.session_state.retriever, "document_search",
            "Searches and returns info from the user's uploaded documents."
        )
        tools = [document_tool] + tools
    
    llm, _ = get_models()
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.info("Ask me about your documents or anything from the web!")

# --- CHAT LOGIC ---
user_question = st.text_input("What would you like to research?")
if user_question:
    try:
        with st.spinner("Thinking..."):
            response = st.session_state.agent_executor.invoke({"input": user_question})
            st.subheader("Answer:")
            st.write(response["output"])
    except Exception as e:
        st.error(f"An error occurred: {e}")