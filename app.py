# app.py

import streamlit as st
from retriever import DocumentRetriever
from generator import ResponseGenerator

# Azure OpenAI Configuration
AZURE_API_KEY = "YOUR_AZURE_API_KEY"
AZURE_ENDPOINT = "https://YOUR_RESOURCE_NAME.openai.azure.com/"
AZURE_DEPLOYMENT_ID = "YOUR_DEPLOYMENT_ID"  # Name of the model deployment

# Initialize retriever and generator
retriever = DocumentRetriever()
retriever.ingest_documents()  # Load all documents
retriever.build_index()  # Build FAISS index

generator = ResponseGenerator(api_key=AZURE_API_KEY, endpoint=AZURE_ENDPOINT, deployment_id=AZURE_DEPLOYMENT_ID)

st.title("Financial Report Chatbot")
st.write("Ask a question about the Annual Reports.")

# User input
query = st.text_input("Your Question")

if query:
    # Retrieve relevant documents
    docs = retriever.retrieve(query)

    # Generate response using retrieved context
    context = " ".join([doc[1] for doc in docs])  # Combine content from top-k docs
    response = generator.generate_response(query, context)

    st.write("### Answer")
    st.write(response)

    st.write("### Citations")
    for doc in docs:
        st.write(f"- From {doc[0]['company']} ({doc[0]['file_name']})")
