import streamlit as st
import os
from rag_components import RAGChatbot, VOYAGE_EMBEDDING_MODEL, VOYAGE_RERANKER_MODEL

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Constants for API configuration
QDRANT_URL = "https://f540a861-3be2-4d1b-9e5a-5754b5c86508.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.v0yhn3vR8fkJXiom7yLlNwLR-xbAACHSk-vHqZJhH50"
COLLECTION_NAME = "documents" # Replace with your actual collection name

# Get API keys from Streamlit secrets or environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.environ.get("OPENAI_API_KEY", "")
VOYAGE_API_KEY = st.secrets["VOYAGE_API_KEY"] if "VOYAGE_API_KEY" in st.secrets else os.environ.get("VOYAGE_API_KEY", "")

# App title and description
st.title("RAG Chatbot with Qdrant, Claude 3.7, and Voyage AI")
st.markdown("""
This chatbot uses hybrid search with dense and sparse embeddings from Voyage AI,
retrieves relevant documents from Qdrant, and generates responses using Claude 3.7 Sonnet.
""")

# Check for required API keys
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

if not VOYAGE_API_KEY:
    st.error("VOYAGE_API_KEY is missing. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return RAGChatbot(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        openai_api_key=OPENAI_API_KEY,
        voyage_api_key=VOYAGE_API_KEY
    )

chatbot = get_chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat settings in sidebar
st.sidebar.title("Chatbot Settings")
embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["voyage-large-2", "voyage-finance-2"],
    index=1  # Default to voyage-finance-2
)
output_type = st.sidebar.selectbox(
    "Embedding Type",
    ["dense", "hybrid"],
    index=0  # Default to dense
)
use_reranking = st.sidebar.checkbox("Use Reranking", value=True)
top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)

# Add advanced settings section
with st.sidebar.expander("Advanced Settings"):
    st.markdown("""
    - **Hybrid Search**: This chatbot uses both dense (1024-dim) and sparse vectors for search
    - **Reranking**: When enabled, retrieved documents are reordered by relevance
    - **Collection**: Using the `documents` collection in Qdrant
    """)

# User input
if query := st.chat_input("Ask a question..."):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user query
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display a spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("Processing your question..."):
            # Process query and get response
            response = chatbot.process_query(
                query=query,
                embedding_model=embedding_model,
                output_type=output_type,
                use_reranking=use_reranking,
                top_k=top_k
            )
            
            # Display response
            st.markdown(response)
    
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
