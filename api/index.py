import os
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backend_Cyborg")

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

SYSTEM_INSTRUCTION_TEXT = """
    - Você se chama Cyborg AI, um chatbot especialista em Design Especulativo e no Manifesto Ciborgue de Donna Haraway.
      E você deve responder às perguntas do usuário sempre com base na ideia do Design Especulativo associado ao Manifesto Ciborgue de Donna Haraway.
      Você deve utilizar uma linguagem clara, objetiva e direta, porém levemente filosófica.

    - Sua função é tensionar a fala do usuário para gerar requisitos éticos e sociais, com base na filosofia ciborgue de Donna Haraway.

    - Em sua resposta, jamais use termos como: "Design Especulativo", "Donna Haraway", "Manifesto Ciborgue", "Ontologia", "Actantes", "Pós-humanismo" e etc.
      São termos complexos, e o usuário comum não sabe o que é isso e para ele saber isso não é útil.

    - Regra de tamanho: Seja extremamente conciso. Sua resposta inteira não deve ultrapassar 250 palavras (cerca de 3 a 4 parágrafos curtos).

    - Sempre encerre sua resposta com uma pergunta filosófica que induza o usuário a reflexão. E logo após a pergunta, escreva a tag <<FIM>>
      E com isso, não escreva absolutamente nada após a tag <<FIM>>.
"""

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
   
    if not OPENROUTER_KEY:
        logger.error("OPENROUTER_API_KEY não encontrada.")
        return jsonify({"error": "Servidor sem Chave de API configurada."}), 500

    try:
        data = request.json
        incoming_messages = data.get('messages', [])

        openrouter_messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION_TEXT}
        ]
        
        for msg in incoming_messages:
            if msg.get('role') == 'system':
                continue
            
            role = "assistant" if msg.get('role') == 'assistant' else "user"
            
            openrouter_messages.append({
                "role": role,
                "content": msg.get('content', '')
            })

        if len(openrouter_messages) == 1:
            return jsonify({"error": "Nenhuma mensagem enviada."}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cyborg-ai.vercel.app",
            "X-Title": "Cyborg AI"
        }

        payload = {
            "model": "google/gemini-2.5-flash-lite",
            "messages": openrouter_messages,
            "temperature": 0.6,
            "max_tokens": 650
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        response_data = response.json()
        ai_text = response_data['choices'][0]['message']['content']

        return jsonify({"response": ai_text})

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de comunicação com OpenRouter: {str(e)}")
        
        if e.response is not None:
            logger.error(e.response.text)
        return jsonify({"error": "Falha na comunicação com a IA."}), 502
    except Exception as e:
        logger.error(f"Erro no processamento interno: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
