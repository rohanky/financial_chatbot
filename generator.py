from openai import AzureOpenAI
import os

openai_client_generator = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class Generator:
    def __init__(self):
        self.model = "gpt-4o"  # Adjust if needed

    def generate_response(self, query, documents):
        """
        Generate response using Azure OpenAI by combining query and top documents.
        """
        context = "\n".join([doc['Content'][:1000] for doc in documents])  # Limit context

        prompt = (
            f"Answer the question based on the context below:\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        response = openai_client_generator.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
