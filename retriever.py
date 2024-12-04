import pandas as pd
import faiss
import numpy as np
import os
from openai import AzureOpenAI

EMBEDDING_MODEL = "text-embedding-3-large"  # Adjust if necessary

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


class Retriever:
    def __init__(self, index_file=None, data_file=None):
        """
        Initialize the retriever by loading data and FAISS index.
        """
        self.embedding_dimension = 768  # Adjust based on your embedding model
        self.data = self.load_and_group_data(data_file)
        self.index = self.load_or_create_index(index_file)

    def load_and_group_data(self, file_path):
        """
        Load and group data by Company, File Name, and Page Number.
        """
        df = pd.read_excel(file_path)
        grouped_df = (
            df.groupby(['Company', 'File Name', 'Page Number'], as_index=False)
            .agg({'Content': lambda x: ' '.join(x.dropna().astype(str))})
        )
        return grouped_df

    def generate_embeddings(self, texts):
        """
        Generate text embeddings using Azure OpenAI Embedding API.
        """
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        # Extract embeddings from the response
        embeddings = np.array([embedding.embedding for embedding in response.data])
        return embeddings

    def generate_embedding_for_row(self, text):
        """
        Generate embeddings for a single row's content.

        Args:
        - text (str): The text content for which to generate embeddings.

        Returns:
        - np.ndarray: The generated embedding as a NumPy array.
        """
        if not isinstance(text, str) or text.strip() == "":
            # Return a zero vector for invalid or empty content
            return np.zeros((self.embedding_dimension,))
        return self.generate_embeddings([text])[0]

    def build_faiss_index(self):
        """
        Build FAISS index from grouped data.
        """
        # Generate embeddings for each row in 'Content'
        self.data['Embeddings'] = self.data['Content'].apply(self.generate_embedding_for_row)

        # Combine embeddings into a single NumPy array
        embeddings = np.vstack(self.data['Embeddings'].tolist())
        dimension = embeddings.shape[1]

        # Build the FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def load_or_create_index(self, index_file):
        """
        Load an existing FAISS index or create a new one.
        """
        if index_file and os.path.exists(index_file):
            index = faiss.read_index(index_file)
        else:
            index = self.build_faiss_index()
        return index

    def save_faiss_index(self, index_file):
        """
        Save FAISS index to a file.
        Ensure that the directory exists before writing the file.
        """
        # Ensure the directory exists
        directory = os.path.dirname(index_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        faiss.write_index(self.index, index_file)

    def search(self, query, top_k=5):
        """
        Search the FAISS index for the top K most relevant documents.
        """
        # Generate embedding for the query
        query_embedding = self.generate_embeddings([query])
        
        # Perform the search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # Skip if no match found
                continue
            result = {
                "Company": self.data.iloc[idx]['Company'],
                "File Name": self.data.iloc[idx]['File Name'],
                "Page Number": self.data.iloc[idx]['Page Number'],
                "Content": self.data.iloc[idx]['Content'],
                "Distance": dist
            }
            results.append(result)
        return results
