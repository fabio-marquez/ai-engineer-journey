import requests
from datetime import datetime
from models import LLMClient

client = LLMClient(model="openai/gpt-oss-20b")

def analisa_conteudo(job_description):

    system_prompt = """
        Você é um experiente analista de job descriptions e sua função é analisar de uma lista de posições quais delas estão relacionadas
        a engenharia de dados ou engenharia de IA.
    """

    user_prompt = f"""
        Analise essa posição e defina se ela é ou não relacionada a engenharia de dados ou engenharia de IA: {job_description}.
        Responda apenas com 'sim' ou 'não'.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    retorno = client.completions(messages)
    return "sim" in retorno.lower()

def main():
    url = "https://remoteok.com/api"
    request = requests.get(url)

    response = request.json()

    last_updated_epoch = response[0]['last_updated']
    last_updated_dt = datetime.fromtimestamp(last_updated_epoch)

    print(f"Dados atualizados em: {last_updated_dt}")

    jobs_relevantes = []
    for posicao in response[1:]:
        if analisa_conteudo(posicao['description']):
            jobs_relevantes.append(posicao)
            print(f"✓ Job relevante encontrado: {posicao['position']}")

    # Salvar jobs relevantes
    print(f"\nTotal de jobs relevantes: {len(jobs_relevantes)}")

if __name__ == "__main__":
    main()