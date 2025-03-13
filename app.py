import streamlit as st
import os
from rag_components import RAGChatbot

# Set page configuration
st.set_page_config(page_title="Finance RAG Chatbot", page_icon="💹", layout="wide")

# Constants for API configuration
QDRANT_URL = "https://f540a861-3be2-4d1b-9e5a-5754b5c86508.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.v0yhn3vR8fkJXiom7yLlNwLR-xbAACHSk-vHqZJhH50"
COLLECTION_NAME = "documents"  # Replace with your actual collection name

# Default models (can be overridden in UI)
DEFAULT_EMBEDDING_MODEL = "voyage-finance-2"  # Finance-specific model
DEFAULT_RERANKER_MODEL = "rerank-2"  # Latest reranker model

# Get API keys from Streamlit secrets or environment variables
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"] if "CLAUDE_API_KEY" in st.secrets else os.environ.get("CLAUDE_API_KEY", "")
VOYAGE_API_KEY = st.secrets["VOYAGE_API_KEY"] if "VOYAGE_API_KEY" in st.secrets else os.environ.get("VOYAGE_API_KEY", "")

# App title and description
st.title("Finance RAG Chatbot")
st.markdown("""
This chatbot uses Voyage AI embeddings and reranking with Qdrant vector search to find relevant financial documents,
and generates responses using Claude 3.7 Sonnet.
""")

# Check for required API keys
if not CLAUDE_API_KEY:
    st.error("CLAUDE_API_KEY is missing. Please set it in Streamlit secrets or as an environment variable.")
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
        claude_api_key=CLAUDE_API_KEY,
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

# Model settings
embedding_model = st.sidebar.selectbox(
    "Embedding Model", 
    ["voyage-finance-2", "voyage-large-2", "voyage-code-2"],
    index=0
)

output_type = st.sidebar.selectbox(
    "Embedding Type", 
    ["dense", "hybrid"],
    index=0,
    help="Finance models may only support dense embeddings"
)

use_reranking = st.sidebar.checkbox("Use Reranking", value=True)

reranker_model = st.sidebar.selectbox(
    "Reranker Model",
    ["rerank-2", "rerank-2-lite"],
    index=0
)

top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)

# Add advanced settings section
with st.sidebar.expander("Advanced Settings"):
    st.markdown("""
    - **voyage-finance-2**: Specialized for financial text and documents
    - **voyage-large-2**: General-purpose model (supports hybrid embeddings)
    - **voyage-code-2**: Optimized for code and technical content
    - **rerank-2**: High-quality reranker with up to 16K tokens
    - **rerank-2-lite**: Faster reranker with up to 8K tokens
    """)

# User input
if query := st.chat_input("Ask a question about finance..."):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user query
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display a spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("Researching financial information..."):
            # Process query and get response
            response = chatbot.process_query(
                query=query,
                embedding_model=embedding_model,
                output_type=output_type,
                use_reranking=use_reranking,
                reranker_model=reranker_model,
                top_k=top_k
            )
            
            # Display response
            st.markdown(response)
    
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
