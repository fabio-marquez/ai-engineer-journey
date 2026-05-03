from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

# Configurações de ambiente
load_dotenv()
secret = os.getenv('GROQ_API_KEY')

# Definição do modelo a ser utilizado
MODEL = "openai/gpt-oss-20b"

# Criando uma conexão ao cliente OpenAI mas usando a base URL da Groq
client = OpenAI(api_key=secret, base_url="https://api.groq.com/openai/v1")

# Configurando as chamadas a LLM
system_prompt = """
Você é um especialista em extrair conteúdo técnico relevante de postagens de emprego na área da tecnologia,
focando na área de inteligência artificial. Sua função é a partir de uma descrição de uma vaga, identificar quais são os 
requisitos mais importantes para que um candidato seja considerado apto para aquela vaga.
Você deve extrair os requisitos técnicos mais relevantes e listá-los da seguinte forma, em um json estruturado:
{"requisitos": [
    {"requisito": "RAG", "importancia": "Alta"},
    {"requisito": "System Design", "importancia": "Média"},
    ...
}.
Você deve classificar a importância de cada requisito como "Alta", "Média" ou "Baixa", com base na frequência e ênfase dada a eles na descrição da vaga.
Além disso, você deve capturar o cerne daquele requisito, não criando longas descrições para cada requisito.
"""

def get_user_prompt(nome_vaga, job_description):
    return f"""
    Aqui está a descrição de uma vaga de emprego para {nome_vaga}: {job_description}
    """

def run_analysis(df):
    respostas = []
    for index, row in df.iterrows():
        message = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":get_user_prompt(row['Posicao'], row['Descricao'])}
            ]
        )
        resposta = message.choices[0].message.content
        respostas.append(resposta)
        #print(f"Requisitos para a vaga {row['Posicao']}: {resposta}")
        return respostas

def summarize_requirements(requirements):

    system_prompt = """
        Você é um especialista em análise de dados e tem a tarefa de analisar uma lista de requisitos técnicos extraídos 
        de várias descrições de vagas de emprego na área de inteligência artificial. Seu objetivo é identificar quais são os 
        requisitos mais comuns e relevantes para os candidatos que buscam oportunidades nessa área.
    """

    user_prompt = f"""
        Aqui estão os requisitos técnicos extraídos de várias descrições de vagas de emprego na área de inteligência artificial: 
        {requirements}
    """
    message = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ]
    )
    resposta = message.choices[0].message.content

    return resposta

def main():
    
    df = pd.read_csv("vagas.csv", delimiter=";")
    
    requisitos = run_analysis(df)

    resultado = summarize_requirements(requisitos)

    with open("requisitos_sintetizados.md", "w") as f:
        f.write(resultado)


if __name__ == "__main__":
    main()
