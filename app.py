import streamlit as st
from retriever import Retriever
from generator import Generator
import os

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

def reset_conversation_history():
    """
    Reset the conversation history when slider value changes.
    """
    st.session_state['conversation_history'] = []

# Constants
PARSED_DATA_FILE = 'parsed_data.xlsx'
INDEX_FILE = 'faiss_index/faiss_index.index'

# Streamlit Page Configuration
st.set_page_config(page_title="Financial Chatbot", layout="wide")

# Title and Description
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
    # Allow the user to adjust the FAISS-TFIDF weight using a float slider
    faiss_weight = st.slider(
        "🔄 Adjust FAISS-TFIDF weight",
        min_value=0.0,  # Float minimum value
        max_value=1.0,  # Float maximum value
        value=0.5,  # Default slider value
        step=0.05,  # Step increment
        key="slider", 
        on_change=reset_conversation_history  # Reset conversation history on change
    )

    # Display current conversation history
    st.write("Conversation History:", st.session_state['conversation_history'])

    # Add to conversation history (for testing purposes)
    if st.button("Add to Conversation"):
        st.session_state['conversation_history'].append(f"Message at slider {faiss_weight}")
        st.write("Conversation History Updated:", st.session_state['conversation_history'])

    # Display search results
    st.write("🔍 Searching relevant documents...")
    results = retriever.hybrid_search(query, top_k=10, faiss_weight=faiss_weight)

    # Generate response
    st.write("🧠 Generating response...")
    response = generator.generate_response(query, results, st.session_state.conversation_history)
    st.success(response)  # Display response in a success box

    # Display retrieved results (top 5 contexts)
    st.write(f"### Found {len(results)} relevant documents (Top 5 shown):")
    for idx, result in enumerate(results[:]):
        st.write(f"**Document {idx + 1}**")
        st.write(f"**Company**: {result['Company']}")
        st.write(f"**File Name**: {result['File Name']}")
        st.write(f"**Page Number**: {result['Page Number']}")
        st.write(f"**Content**: {result['Content'][:500]}...")  # Limit content for display
        
        # Display the source and scores
        st.write(f"**Source**: {'Semantic (FAISS)' if result.get('FAISS Score') is not None else 'Lexical (TF-IDF)'}")
        st.write(f"**FAISS Score**: {result['FAISS Score']:.4f}")  # Display FAISS Score
        st.write(f"**TF-IDF Similarity**: {result['TF-IDF Similarity']:.4f}")
        st.write(f"**Combined Score**: {result['Combined Score']:.4f}")
        st.write("---")

    # Save conversation history
    st.session_state.conversation_history.append({"query": query, "response": response})
