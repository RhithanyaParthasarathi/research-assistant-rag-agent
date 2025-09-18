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
from langchain_core.messages import HumanMessage, AIMessage

# Qdrant client
from qdrant_client import QdrantClient, models

# Load environment variables
load_dotenv()

# --- SETUP ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# --- SESSION STATE INITIALIZATION ---
# All state variables are now managed in one place.
def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'agent_executor' not in st.session_state:
        st.session_state.agent_executor = None
    if 'processed_file_names' not in st.session_state:
        st.session_state.processed_file_names = []
    # NEW: Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Call the initialization function at the start
initialize_session_state()

# --- CORE CACHED FUNCTIONS ---
@st.cache_resource
def get_llm():
    """Initializes and returns the LLM."""
    # Corrected the model name to a valid one
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_embeddings():
    """Initializes and returns the local embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    embeddings = get_embeddings()
    texts = [doc.page_content for doc in splits]
    metadatas = [dict(doc.metadata, tenant=session_id) for doc in splits]

    QdrantVectorStore.from_texts(
        texts=texts, embedding=embeddings, metadatas=metadatas, url=QDRANT_URL,
        api_key=QDRANT_API_KEY, collection_name=COLLECTION_NAME, force_recreate=False
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.tenant",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=COLLECTION_NAME
    )
    return vector_store.as_retriever(
        search_kwargs={'filter': models.Filter(must=[
            models.FieldCondition(key="metadata.tenant", match=models.MatchValue(value=session_id))
        ])}
    )

# --- AGENT AND TOOL CREATION ---
def create_agent_executor():
    """Creates the agent executor with tools."""
    llm = get_llm()
    tools = [DuckDuckGoSearchRun(name="web_search")]

    if st.session_state.retriever:
        document_tool = create_retriever_tool(
            st.session_state.retriever,
            "document_search",
            "Searches and returns info from the user's uploaded documents. Use this tool for any questions about the documents."
        )
        tools.append(document_tool)
    
    # NEW: Modify the prompt to include chat history
    prompt = hub.pull("hwchase17/react-chat")
    # The new prompt template includes a `chat_history` placeholder
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms to the expected format."
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

# --- DOCUMENT PROCESSING AND AGENT RE-INITIALIZATION ---
current_file_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []
if uploaded_files and current_file_names != st.session_state.processed_file_names:
    with st.spinner("Processing documents..."):
        st.session_state.retriever = ingest_and_create_retriever(uploaded_files, st.session_state.session_id)
        st.session_state.processed_file_names = current_file_names
        # Important: Re-create the agent executor so it gets the new document_search tool
        st.session_state.agent_executor = create_agent_executor()
    st.success("Documents processed successfully!")

# Create the agent executor if it doesn't exist
if not st.session_state.agent_executor:
    st.session_state.agent_executor = create_agent_executor()

# --- CHAT UI LOGIC (REVISED AND FIXED) ---
st.info("Ask me about your documents or anything from the web!")

# Display existing messages from history on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to research?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in a new message bubble
    with st.chat_message("assistant"):
        # Create a placeholder for the response
        message_placeholder = st.empty()
        # Display a thinking message immediately
        message_placeholder.markdown("Thinking...")

        try:
            # Prepare chat history for the agent
            chat_history = [
                HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content'])
                for msg in st.session_state.messages[:-1] # All messages except the new user prompt
            ]

            # Invoke the agent to get the response
            response = st.session_state.agent_executor.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            
            # THE FIX for "undefined": Safely get the output. If it's None, provide a default.
            ai_response = response.get("output")
            if ai_response is None:
                ai_response = "I'm sorry, I couldn't formulate a response."

        except Exception as e:
            ai_response = f"An error occurred: {e}"
        
        # Update the placeholder with the final response
        message_placeholder.markdown(ai_response)
        
        # Add the final AI response to the message history for the next run
        st.session_state.messages.append({"role": "assistant", "content": ai_response})