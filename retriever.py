import pandas as pd
import faiss
import numpy as np
import os
from openai import AzureOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        Initialize the retriever by loading data, FAISS index, and TF-IDF matrix.
        """
        self.embedding_dimension = 768  # Adjust based on your embedding model
        self.data = self.load_and_group_data(data_file)
        
        # Custom stop words for financial documents (optional)

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data['Content'])

        self.index = self.load_or_create_index(index_file)

    def load_and_group_data(self, file_path, split_length=1000):
        """
        Load and group data by Company, File Name, and Page Number.
        Split content if it exceeds a certain length.
        """
        df = pd.read_excel(file_path)
        grouped_df = (
            df.groupby(['Company', 'File Name', 'Page Number'], as_index=False)
            .agg({'Content': lambda x: ' '.join(x.dropna().astype(str))})
        )

        # Split content if it exceeds the split length
        split_rows = []
        for _, row in grouped_df.iterrows():
            content = row['Content']
            if len(content) > split_length:
                parts = [content[i:i+split_length] for i in range(0, len(content), split_length)]
                for idx, part in enumerate(parts):
                    split_rows.append({
                        'Company': row['Company'],
                        'File Name': row['File Name'],
                        'Page Number': f"{row['Page Number']} - Part {idx + 1}",
                        'Content': part
                    })
            else:
                split_rows.append(row)

        return pd.DataFrame(split_rows)

    def generate_embeddings(self, texts):
        """
        Generate text embeddings using Azure OpenAI Embedding API.
        """
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        embeddings = np.array([embedding.embedding for embedding in response.data])
        return embeddings

    def generate_embedding_for_row(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return np.zeros((self.embedding_dimension,))
        return self.generate_embeddings([text])[0]

    def build_faiss_index(self):
        """
        Build FAISS index from grouped data.
        """
        self.data['full_content'] = self.data['File Name'] + " " + self.data['Content']
        self.data['Embeddings'] = self.data['full_content'].apply(self.generate_embedding_for_row)

        embeddings = np.vstack(self.data['Embeddings'].tolist())
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def load_or_create_index(self, index_file):
        if index_file and os.path.exists(index_file):
            return faiss.read_index(index_file)
        else:
            return self.build_faiss_index()

    def save_faiss_index(self, index_file):
        directory = os.path.dirname(index_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        faiss.write_index(self.index, index_file)

    def faiss_search(self, query, top_k=5):
        """
        Perform semantic search using FAISS.
        """
        query_embedding = self.generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append({
                "Company": self.data.iloc[idx]['Company'],
                "File Name": self.data.iloc[idx]['File Name'],
                "Page Number": self.data.iloc[idx]['Page Number'],
                "Content": self.data.iloc[idx]['Content'],
                "FAISS Score": 1 - dist  # Convert distance to similarity
            })
        return results

    def tfidf_search(self, query, top_k=5):
        """
        Perform lexical search using TF-IDF.
        """
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "Company": self.data.iloc[idx]['Company'],
                "File Name": self.data.iloc[idx]['File Name'],
                "Page Number": self.data.iloc[idx]['Page Number'],
                "Content": self.data.iloc[idx]['Content'],
                "TF-IDF Similarity": similarities[idx]
            })
        return results

    def hybrid_search(self, query, top_k=10, faiss_weight=0.5):
        """
        Perform a hybrid search combining FAISS and TF-IDF scores.
        """
        faiss_results = self.faiss_search(query, top_k)
        tfidf_results = self.tfidf_search(query, top_k)

        combined_results = {}

        for res in faiss_results:
            key = (res['Company'], res['File Name'], res['Page Number'])
            combined_results[key] = {
                "Company": res['Company'],
                "File Name": res['File Name'],
                "Page Number": res['Page Number'],
                "Content": res['Content'],
                "FAISS Score": res['FAISS Score'],
                "TF-IDF Similarity": 0,
            }

        for res in tfidf_results:
            key = (res['Company'], res['File Name'], res['Page Number'])
            if key not in combined_results:
                combined_results[key] = {
                    "Company": res['Company'],
                    "File Name": res['File Name'],
                    "Page Number": res['Page Number'],
                    "Content": res['Content'],
                    "FAISS Score": 0,
                    "TF-IDF Similarity": res['TF-IDF Similarity'],
                }
            else:
                combined_results[key]["TF-IDF Similarity"] = res['TF-IDF Similarity']

        # Calculate combined scores
        for result in combined_results.values():
            result["Combined Score"] = (
                faiss_weight * result["FAISS Score"] + (1 - faiss_weight) * result["TF-IDF Similarity"]
            )

        sorted_results = sorted(combined_results.values(), key=lambda x: x['Combined Score'], reverse=True)
        return sorted_results[:top_k]
