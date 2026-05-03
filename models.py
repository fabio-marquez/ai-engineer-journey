from dotenv import load_dotenv
import os
from openai import OpenAI

# Configurações de ambiente
load_dotenv()
secret = os.getenv('GROQ_API_KEY')

class LLMClient:
    def __init__(self, api_key=secret, base_url="https://api.groq.com/openai/v1", model="llama3-70b-8192"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def completions(self, messages):

        resposta = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return resposta.choices[0].message.content
