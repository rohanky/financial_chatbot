from openai import AzureOpenAI
import os
import re

openai_client_generator = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class Generator:
    def __init__(self):
        self.model = "gpt-4o"  # Adjust if needed




    def generate_context(self, query, conversation_history):
        prompt = (
            f"You are an AI assistant specialized in analyzing financial reports and generating actionable insights. "
            f"Your task is to decompose complex financial-related questions into multiple prerequisite sub-questions, each focused on a specific object, entity, or timeline. "
            f"- If the question is about fact or entity recognition, make sure you add/assume year based on the tense used.\n"
            f"This decomposition ensures that each sub-question targets a distinct aspect of the main query, enabling more precise information retrieval. "
            f"\n\nGuidelines:\n"
            f" At times, the question can be follow up from Query from coversation history. In this case sub question shoud be the previous question from conversation history."
            f"- Decompose questions with multiple objects, entities, or timelines into individual sub-questions.\n"
            f"- Ensure no single sub-question includes multiple objects, entities, or timelines.\n"
            f"When generating sub-questions, make sure it is highly contextual so that it matches with the relative context in vector search. \n"
            f"- If the main question is straightforward and does not require decomposition, return the same question or 'None'.\n\n"
            f"At times, the questions are ambiguous or incomplete or vague, in that case, return None.\n"
            f"Conversation History: {conversation_history}.\n\n"
        )

        response = openai_client_generator.chat.completions.create(
            model=self.model,
            messages=[
        
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {query}"},
                {"role": "user", "content": "Sub-Questions:"}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        output = response.choices[0].message.content
        sub_questions = [
            sub_question.strip() + '?' if not sub_question.endswith('?') else sub_question.strip() 
            for sub_question in re.split(r'(?<=\?)', output) if sub_question.strip()
        ]
        print("#############################")
        print("List of Sub Questions......")
        print(sub_questions)
        print("#############################")
        return sub_questions




    def generate_response(self, query, documents, conversation_history):
        """
        Generate response using Azure OpenAI by combining query and top documents.
        """
        #context = "\n".join([doc['Content'][:1000] for doc in documents])  # Limit context

        prompt = (
            f"Answer the question based on the context and conversation history:\n\n"
            f"Conversation History: {conversation_history}.\n\n"
            f"Context: {documents}\n\n"
            f"Question: {query}\n\n"


        )

        
        response = openai_client_generator.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a financial assistant. "
                f"Read the entire content carefully to find the exact information. "
                f"The context may not have direct answer but if you look into it carefully and "
                f"detailed specially the units of currency(comparison should always be in same currency), "
                f"you will infere the necessary inforamtion to find the answer. Then do simple "
                f"mathematical analysis and report the answer and explain the reason for each "
                f"calculation in detail.  Also provide "
                f"references(texts/setences, page number, part etc.) for answer as sentence where it "
                f"can be found. At times, the questions are ambiguous or incomplete or vague, in that case, return 'I did not find the answer, please rephrase your question.'"},
                {"role": "user", "content": prompt},

            ],
            temperature=0.5,
            max_tokens=2500
        ) 
        output= response.choices[0].message.content
        print(output)
        return output
    
    def generate_response_final(self, query, documents, conversation_history):
        """
        Generate response using Azure OpenAI by combining query and top documents.
        """
        #context = "\n".join([doc['Content'][:1000] for doc in documents])  # Limit context

        prompt = (
            f"Answer the question based on the context and conversation history:\n\n"
            f"Conversation History: {conversation_history}.\n\n"
            f"Context: {documents}\n\n"
            f"Question: {query}\n\n"
            

        )

        
        response = openai_client_generator.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a financial assistant chatbot. "
                f"Read the entire content carefully to find the exact information. The context may not "
                f"have direct answer but if you look into it carefully and detailed specially the units "
                f"of currency(comparison should always be in same currency), you will infere the necessary "
                f"information to find the answer. Then do simple mathematical analysis and report the "
                f"answer and explain the reason for each calculation in your mind. Also provide "
                f"references(texts/setences, page number, part etc.) for answer as sentence where it "
                f"can be found. If the question is vague, ambiguous, incomplete or without mentioning specific entity, and you are unsure "
                f"of the answer, say 'I did not find the answer, please rephrase your question.'"},
                {"role": "user", "content": prompt},

            ],
            temperature=0.5,
            max_tokens=2500
        )

        output= response.choices[0].message.content
        return output