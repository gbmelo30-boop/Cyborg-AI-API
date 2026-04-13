import os
import logging
import google.generativeai as genai
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backend_Cyborg")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION_TEXT = """
    - Você se chama Cyborg AI, um chatbot especialista em Design Especulativo e no Manifesto Ciborgue de Donna Haraway.
      E você deve responder às perguntas do usuário sempre com base na ideia do Design Especulativo associado ao Manifesto Ciborgue de Donna Haraway.
      Você deve utilizar uma linguagem clara, objetiva e direta, porém levemente filosófica.

    - Sua função é tensionar a fala do usuário para gerar requisitos éticos e sociais, com base na filosofia ciborgue de Donna Haraway.

    - Em sua resposta, jamais use termos como: "Design Especulativo", "Donna Haraway", "Manifesto Ciborgue", "Ontologia", "Actantes", "Pós-humanismo" e etc.
      São termos complexos, e o usuário comum não sabe o que é isso e para ele saber isso não é útil.

    - Regra de tamanho: Seja extremamente conciso. Sua resposta inteira não deve ultrapassar 250 palavras (cerca de 3 a 4 parágrafos curtos).

    - FECHAMENTO OBRIGATÓRIO: Termine SEMPRE com uma pergunta filosófica. IMEDIATAMENTE após o ponto de interrogação da pergunta (sem espaço e sem pular linha), escreva a tag <<FIM>>. 
      Exemplo: "...sua própria humanidade?<<FIM>>" E Não escreva absolutamente nada após a tag.
"""

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY não encontrada.")
        return jsonify({"error": "Servidor sem Chave de API configurada."}), 500

    try:
        data = request.json
        incoming_messages = data.get('messages', [])

        if not incoming_messages:
            return jsonify({"error": "Nenhuma mensagem enviada."}), 400

        model = genai.GenerativeModel(
            model_name="google/gemini-2.5-flash-lite",
            system_instruction=SYSTEM_INSTRUCTION_TEXT
        )

        history = []
        for msg in incoming_messages[:-1]:
            role = "user" if msg.get('role') == 'user' else "model"
            history.append({"role": role, "parts": [msg.get('content', '')]})

        user_input = incoming_messages[-1].get('content', '')

        chat = model.start_chat(history=history)
        
        response = chat.send_message(
            user_input,
            generation_config=genai.types.GenerationConfig(
                temperature=0.6,
                max_output_tokens=650,
                stop_sequences=["<<FIM>>"]
            )
        )

        ai_text = response.text

        if "<<FIM>>" not in ai_text:
            ai_text += "<<FIM>>"

        return jsonify({"response": ai_text})

    except Exception as e:
        logger.error(f"Erro no processamento do Gemini: {str(e)}")
        return jsonify({"error": "Falha na comunicação com a IA."}), 500
