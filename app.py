import streamlit as st
from retriever import Retriever
from generator import Generator
import os

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

def reset_conversation_history():
    """
    Reset the conversation history and extended context when slider value changes.
    """
    st.session_state['conversation_history'] = []
    st.session_state['extended_context'] = []

# Constants
PARSED_DATA_FILE = 'parsed_data.xlsx'
INDEX_FILE = 'faiss_index/faiss_index.index'

# Streamlit Page Configuration
st.set_page_config(page_title="Financial Chatbot", layout="wide")

# Title and Description
st.write("### 📊 Financial Chatbot")
st.write("This chatbot helps you answer questions based on detailed financial reports from companies like Crayon, SoftwareOne, and Uber. Ask questions about specific financial metrics, revenue, growth, and more!")

# Load FAISS index and data
@st.cache_resource
def load_retriever_and_generator():
    """
    Load or initialize the retriever and generator.
    """
    if os.path.exists(INDEX_FILE):
        st.write("🔄 Loading FAISS index...")
        retriever = Retriever(index_file=INDEX_FILE, data_file=PARSED_DATA_FILE)
    else:
        st.write("🛠️ Building FAISS index from scratch...")
        retriever = Retriever(data_file=PARSED_DATA_FILE)
        retriever.save_faiss_index(INDEX_FILE)

    generator = Generator()
    return retriever, generator

retriever, generator = load_retriever_and_generator()

# User Query Input
query = st.text_input("💬 Enter your financial query:")

if query:
    # Step 1: Refine the query
    refined_questions = generator.generate_context(query,st.session_state.conversation_history)


    # Step 2: Retrieve and process results
    extended_context = []
    faiss_weight = 1

    # Reset conversation history and context when the slider is moved
    reset_conversation_history()

    if 'none' in refined_questions[0].lower():
        st.write('⚡ Generating quick response...')
        results = retriever.hybrid_search(query, top_k=10, faiss_weight=faiss_weight)  # Use slider value for weight
        extended_context.append(results)
    else:
        st.write('🔍 Analyzing in depth...This might take a while...')
        # Process refined questions and use the slider weight
        for sub_query in refined_questions:
            
            #st.write(f"Retrieving results for: {sub_query}")
            results = retriever.hybrid_search(sub_query, top_k=10, faiss_weight=faiss_weight)  # Use slider value for weight
            # Generate response
            #st.write("🧠 Generating response...")  
            response = generator.generate_response(sub_query, results, st.session_state.conversation_history)
            #st.success(response)  # Display response in a success box

            extended_context.append({
                "question": sub_query,
                "context": results,
                "answer": response
            })



    final_answer = generator.generate_response_final(query, extended_context, st.session_state.conversation_history)
    st.success(final_answer) 

    st.write("### Extended Context (optional):")
    with st.expander("Click to expand extended context"):
        st.write(extended_context)
    # Save conversation history
    st.session_state['conversation_history'].append({
        "query": query,
        "Context": extended_context,
        "response": final_answer
    })
