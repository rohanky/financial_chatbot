# retriever.py

import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    def __init__(self, data_path='data/', model_name='all-MiniLM-L6-v2'):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs = []

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file."""
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def extract_text_from_html(self, file_path):
        """Extract text from an HTML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text()
        return text

    def ingest_documents(self):
        """Extract text from all PDF and HTML files and store them."""
        for company in os.listdir(self.data_path):
            company_path = os.path.join(self.data_path, company)
            if os.path.isdir(company_path):
                for file in os.listdir(company_path):
                    file_path = os.path.join(company_path, file)
                    if file.endswith('.pdf'):
                        text = self.extract_text_from_pdf(file_path)
                    elif file.endswith('.html'):
                        text = self.extract_text_from_html(file_path)
                    else:
                        continue

                    # Store the text along with the file name and company
                    self.docs.append({"company": company, "file_name": file, "content": text})

    def build_index(self):
        """Create a FAISS index from the document embeddings."""
        embeddings = []
        for doc in self.docs:
            doc_embedding = self.model.encode(doc['content'])
            embeddings.append(doc_embedding)

        # Build FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))

    def retrieve(self, query, top_k=3):
        """Retrieve top-k documents for a given query."""
        query_embedding = self.model.encode(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.docs[idx][['company', 'file_name']], self.docs[idx]['content']) for idx in indices[0]]

# Example Usage:
# retriever = DocumentRetriever()
# retriever.ingest_documents()
# retriever.build_index()
# retriever.retrieve("What is the revenue for 2023?")
