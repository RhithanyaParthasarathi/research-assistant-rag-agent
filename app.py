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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain import hub

# Qdrant client
from qdrant_client.http import models as rest

# Load environment variables
load_dotenv()

# --- SETUP ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "demo_collection_multitenant" # Using a new collection name

# --- SESSION STATE MANAGEMENT ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- CORE FUNCTIONS ---
@st.cache_resource
def get_llm_and_embeddings():
    """Initializes and returns the LLM and embedding models."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return llm, embeddings

def get_agent_executor(tools):
    """Creates and returns the agent executor."""
    llm, _ = get_llm_and_embeddings()
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def ingest_and_get_retriever(uploaded_files, session_id):
    """Ingests uploaded files into Qdrant with a tenant ID and returns a retriever for that tenant."""
    if not uploaded_files:
        return None

    docs = []
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif uploaded_file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_path)
        else:
            continue
        docs.extend(loader.load())

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    _, embeddings = get_llm_and_embeddings()
    
    # Ingest documents with the session_id as the tenant key
    QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        ids=[str(uuid.uuid4()) for _ in splits],  # Unique IDs for each chunk
        metadatas=[dict(doc.metadata, tenant=session_id) for doc in splits] # Add tenant ID to metadata
    )

    # Create a retriever that filters by the current user's session_id
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={'filter': rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.tenant",
                    match=rest.MatchValue(value=session_id),
                )
            ]
        )}
    )
    return retriever

# --- STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="Multi-User AI Research Assistant", page_icon="🔬")
st.title("🔬 AI Research Assistant")

with st.sidebar:
    st.header("Your Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDFs, DOCX, or PPTX files here for this session:",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'pptx']
    )
    st.info(f"Current Session ID: {st.session_state.session_id}")

# Initialize tools
web_search_tool = DuckDuckGoSearchRun()
tools = [web_search_tool]

if uploaded_files:
    with st.spinner("Processing your documents..."):
        retriever = ingest_and_get_retriever(uploaded_files, st.session_state.session_id)
        if retriever:
            document_tool = create_retriever_tool(
                retriever,
                "document_search",
                "Searches and returns information from the user's uploaded documents. Use this for specific questions about the provided context."
            )
            tools = [document_tool, web_search_tool]

st.info("Ask me about your documents or anything from the web!")

try:
    agent_executor = get_agent_executor(tools)
    user_question = st.text_input("What would you like to research?")

    if user_question:
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": user_question})
            st.subheader("Answer:")
            st.write(response["output"])
except Exception as e:
    st.error(f"An error occurred: {e}")
    