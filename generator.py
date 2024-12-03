# generator.py

import openai

class ResponseGenerator:
    def __init__(self, api_key, endpoint, deployment_id):
        openai.api_key = api_key
        openai.api_base = endpoint
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"  # Adjust based on Azure OpenAI version
        self.deployment_id = deployment_id

    def generate_response(self, query, context):
        """Generate a response based on the query and context using Azure OpenAI."""
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = openai.Completion.create(
            engine=self.deployment_id,
            prompt=prompt,
            max_tokens=200
        )
        return response.choices[0].text.strip()
