import streamlit as st
from retriever import Retriever
from generator import Generator
import os

# Constants
PARSED_DATA_FILE = 'parsed_data.xlsx'
INDEX_FILE = 'faiss_index/faiss_index.index'

st.set_page_config(page_title="Financial Chatbot", layout="wide")

# Title and description
st.title("📊 Financial Document Chatbot (RAG)")
st.write("Ask questions about financial documents from Crayon, SoftwareOne, and Uber.")

# Load FAISS index and data
@st.cache_resource
def load_retriever_and_generator():
    """
    Load or initialize the retriever and generator.
    """
    if os.path.exists(INDEX_FILE):
        st.write("Loading FAISS index...")
        retriever = Retriever(index_file=INDEX_FILE, data_file=PARSED_DATA_FILE)
    else:
        st.write("Building FAISS index from scratch...")
        retriever = Retriever(data_file=PARSED_DATA_FILE)
        retriever.save_faiss_index(INDEX_FILE)

    generator = Generator()
    return retriever, generator

retriever, generator = load_retriever_and_generator()

# User Query Input
query = st.text_input("💬 Enter your financial query:")

if query:
    # Display search results
    st.write("🔍 Searching relevant documents...")
    results = retriever.search(query)

     # Generate response
    st.write("🧠 Generating response...")
    response = generator.generate_response(query, results)
    st.success(response)

    # Display retrieved results
    st.write(f"Found {len(results)} relevant documents:")
    for result in results:
        st.write(f"**Company**: {result['Company']}")
        st.write(f"**File Name**: {result['File Name']}")
        st.write(f"**Page Number**: {result['Page Number']}")
        st.write(f"**Content**: {result['Content'][:500]}...")  # Limit content for display
        st.write("---")


