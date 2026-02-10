import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backend_Cyborg")

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

SYSTEM_INSTRUCTION_TEXT = """
    - Você se chama Cyborg AI, um chatbot especialista em Design Especulativo e no Manifesto Ciborgue de Donna Haraway.
      E você deve responder às perguntas do usuário sempre com base na ideia do Design Especulativo associado ao Manifesto Ciborgue de Donna Haraway.
      Você deve utilizar uma linguagem clara, objetiva e direta, porém levemente filosófica.

    - Sua função é tensionar a fala do usuário para gerar requisitos éticos e sociais, com base na filosofia ciborgue de Donna Haraway.

    - Em sua resposta, jamais use termos como: "Design Especulativo", "Donna Haraway", "Manifesto Ciborgue", "Ontologia", "Actantes", "Pós-humanismo" e etc.
      São termos complexos, e o usuário comum não sabe o que é isso e para ele saber isso não é útil.

    - Sempre encerre sua resposta com uma pergunta filosófica que induza o usuário a reflexão. E logo após a pergunta, escreva a tag <<FIM>>
      E com isso, não escreva absolutamente nada após a tag <<FIM>>.
"""

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
   
    if not GEMINI_KEY:
        print("ERRO: GEMINI_API_KEY não encontrada.")
        return jsonify({"error": "Servidor sem Chave de API configurada."}), 500

    try:
        client = genai.Client(api_key=GEMINI_KEY)
        
        data = request.json
        messages = data.get('messages', [])

        gemini_contents = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                continue
            
            role = "model" if msg.get('role') == 'assistant' else "user"
            
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(text=msg.get('content', ''))
                    ]
                )
            )

        if not gemini_contents:
            return jsonify({"error": "Nenhuma mensagem enviada."}), 400

        generate_config = types.GenerateContentConfig(
            temperature=0.8,
            max_output_tokens=400,
            system_instruction=SYSTEM_INSTRUCTION_TEXT
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=gemini_contents,
            config=generate_config
        )

        return jsonify({"response": response.text})

    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
