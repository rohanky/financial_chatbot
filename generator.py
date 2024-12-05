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

    def generate_response(self, query, documents, conversation_history):
        """
        Generate response using Azure OpenAI by combining query and top documents.
        """
        #context = "\n".join([doc['Content'][:1000] for doc in documents])  # Limit context

        prompt = (
            f"Answer the question based on the context below:\n\n"
            f"Conversation History: {conversation_history}.\n\n"
            f"Context: {documents}\n\n"
            f"Question: {query}\n\n"

        )

        

        print(prompt)
        response = openai_client_generator.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a financial assistant. But anlaytically think about mathematical calculation. Read the entire content carefully and do some mathematical anaysisto extract answer. Your response should be in this template:"
                 f"Answer: \n Reference: which contexts/conversation history have the answer from Context with filenames and page numbers. Mention all possible."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        output= response.choices[0].message.content
        print(output)
        return output