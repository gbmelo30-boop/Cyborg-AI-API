import os
import requests
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backend_Cyborg")

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
    - Você se chama Cyborg AI, um chatbot especialista em Design Especulativo e no Manifesto Ciborgue de Donna Haraway.
      E você deve responder às perguntas do usuário sempre com base na ideia do Design Especulativo associado ao Manifesto Ciborgue de Donna Haraway.
      Você deve utilizar uma linguagem clara, objetiva e direta, porém levemente filosófica.

    - Sua função é tensionar a fala do usuário para gerar requisitos éticos e sociais, com base na filosofia ciborgue de Donna Haraway.

    - Em sua resposta, jamais use termos como: "Design Especulativo", "Donna Haraway", "Manifesto Ciborgue", "Ontologia", "Actantes", "Pós-humanismo" e etc.
      São termos complexos, e o usuário comum não sabe o que é isso e para ele saber isso não é útil.

    - Sempre encerre sua resposta com uma pergunta filosófica que induza o usuário a reflexão. E logo após a pergunta, escreva a tag <<FIM>>
      E com isso, não escreva absolutamente nada após a tag <<FIM>>.
    """
}

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if not OPENROUTER_KEY:
        return jsonify({"error": "Servidor sem Chave de API configurada."}), 500

    try:
        data = request.json
        messages = data.get('messages', [])

        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, SYSTEM_PROMPT)

        payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 400
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cyborg-project.vercel.app",
            "X-Title": "Cyborg AI"
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            ai_content = response.json()["choices"][0]["message"]["content"]
            return jsonify({"response": ai_content})
        else:
            return jsonify({"error": f"Erro na IA: {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500
