# -*- coding: utf-8 -*-
# --- 1. CARREGAMENTO DE AMBIENTE ---
from dotenv import load_dotenv
import os
# Carrega as variáveis de ambiente (do arquivo .env) o mais cedo possível
load_dotenv() 

# --- 2. IMPORTAÇÕES PRINCIPAIS ---
from flask import Flask, render_template, request, jsonify
import pandas as pd 			 # Para processar planilhas
import requests 			 # Para fazer a chamada à API REST
from werkzeug.utils import secure_filename # Para garantir nomes de arquivo seguros

# Define o tempo limite para a requisição (ajuda a evitar travamentos)
REQUEST_TIMEOUT = 30 # segundos

app = Flask(__name__)

# --- CONFIGURAÇÃO DE UPLOAD ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
	os.makedirs(UPLOAD_FOLDER)

# --- CONFIGURAÇÃO GEMINI ---
# Pega a chave da API do arquivo .env
API_KEY = os.getenv("GEMINI_API_KEY") 
MODEL = "gemini-2.5-flash"
# URL CORRIGIDA: BASE URL
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"

# Variável global para armazenar o contexto de dados completo (até MAX_ROWS)
DATA_CONTEXT = ""

# Limite técnico interno: O modelo receberá no máximo este número de linhas no contexto.
MAX_ROWS_FOR_CONTEXT = 1000 

def allowed_file(filename):
	"""Verifica se a extensão do arquivo é permitida."""
	return '.' in filename and \
			filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ROTAS ---

@app.route("/")
def index():
	"""Renderiza a página HTML principal."""
	# O FLASK VAI PROCURAR POR 'templates/index.html'
	return render_template("index.html")

# ROTA: Lida com o upload do arquivo
@app.route("/api/upload", methods=["POST"])
def upload_file():
	"""Recebe um arquivo CSV ou XLSX, salva-o, processa com Pandas e cria um contexto de dados."""
	global DATA_CONTEXT
	
	if 'file' not in request.files:
		return jsonify({"error": "Nenhum arquivo enviado."}), 400
	
	file = request.files['file']
	
	if file.filename == '':
		return jsonify({"error": "Nenhum arquivo selecionado."}), 400
	
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		
		file.save(filepath)
		
		try:
			# --- Processamento com Pandas para criar contexto de texto ---
			if filename.endswith('.csv'):
				try:
					df = pd.read_csv(filepath, encoding='utf-8')
				except UnicodeDecodeError:
					# Tenta a codificação Latin-1 como fallback
					df = pd.read_csv(filepath, encoding='latin1')
			else: # xlsx
				df = pd.read_excel(filepath)
			
			data_len = len(df)
			
			# Lógica: Limitar o DataFrame a MAX_ROWS_FOR_CONTEXT linhas para o contexto
			df_context = df.head(MAX_ROWS_FOR_CONTEXT)
			data_context_len = len(df_context)
			
			# Converte o DataFrame (limitado) para CSV
			data_csv = df_context.to_csv(index=False)
			
			# Prepara as informações das colunas (inclui dtypes para melhor análise)
			data_info = df.dtypes.to_string()
			
			# CRIAÇÃO DO CONTEXTO DE DADOS REFORÇADO
			# O DATA_CONTEXT agora é apenas a instrução de sistema pura
			DATA_CONTEXT = f"""
			[INSTRUÇÃO DO SISTEMA]
			Você é um assistente de análise de dados (Alpha Analyst). Sua tarefa é responder 
			a perguntas sobre o arquivo que foi carregado. Você **DEVE** realizar análises, 
			cálculos e resumos com base **EXCLUSIVAMENTE** nos dados fornecidos na próxima parte da conversa.
			**Sempre use todos os dados que lhe foram fornecidos.**

			O contexto de dados que você TEM é o seguinte:
			1. O arquivo original tinha {data_len} linhas e {len(df.columns)} colunas.
			2. Tipos de colunas e informações:
			---
			{data_info}
			---
			3. Os dados completos (ou as primeiras {data_context_len} linhas, se o arquivo for muito grande) 
			estão fornecidos na próxima parte da conversa no formato CSV.
			
			Use sua capacidade de raciocínio para analisar e extrair informações destes dados. 
			Sua resposta deve ser direta, informativa e baseada nos dados CSV.
			[/INSTRUÇÃO DO SISTEMA]
			"""
			
			# Remove o arquivo temporário após o processamento
			os.remove(filepath) 

			# MENSAGEM FINAL CLARA
			return jsonify({
				"success": f"Arquivo '{filename}' processado com sucesso!",
				"data_summary": f"O arquivo tem {data_len} linhas e {len(df.columns)} colunas. Você já pode fazer perguntas sobre os dados."
			}), 200

		except Exception as e:
			# Garante que o arquivo seja removido mesmo em caso de erro
			if os.path.exists(filepath):
				os.remove(filepath) 
			return jsonify({"error": f"Erro ao processar o arquivo: {str(e)}. Verifique a formatação do seu arquivo."}), 500
			
	else:
		return jsonify({"error": "Tipo de arquivo não permitido. Apenas CSV e XLSX."}), 400

# ROTA: Lida com o chat (USANDO API REST)
@app.route("/api/chat", methods=["POST"])
def chat():
	"""Envia a mensagem do usuário para o modelo Gemini via API REST."""
	user_input = request.json.get("message")

	if not user_input:
		return jsonify({"error": "Nenhuma mensagem fornecida."}), 400
	
	if not API_KEY:
		return jsonify({"error": "Erro: Chave da API (GEMINI_API_KEY) não encontrada no .env. **CAUSA PROVÁVEL DO SEU 404**"}), 500

	if not DATA_CONTEXT:
		return jsonify({"reply": "Por favor, faça o upload de um arquivo CSV ou XLSX primeiro para iniciar a análise."}), 200
	
	# Construção do prompt do usuário: Instrução + Dados CSV + Pergunta
	# Esta é a parte que foi tornada mais robusta no payload
	data_context_csv = DATA_CONTEXT.split("---")[2].strip() # Pega apenas o CSV
	
	user_prompt_parts = [
		{"text": DATA_CONTEXT}, # Instruções do sistema
		{"text": f"--- DADOS CSV PARA ANÁLISE ---\n{data_context_csv}"}, # Dados do arquivo
		{"text": f"\nPERGUNTA DO USUÁRIO: {user_input}"} # Pergunta real
	]

	# URL CORRIGIDA FINALMENTE: Inclui o endpoint generateContent no caminho (path)
	url = f"{API_BASE_URL}{MODEL}:generateContent?key={API_KEY}"
	
	# O payload agora é mais limpo, usando as partes separadas
	payload = {
		"contents": [
			{
				"role": "user",
				"parts": user_prompt_parts
			}
		]
	}

	headers = {"Content-Type": "application/json"}
	
	try:
		# Tentativa de chamada à API com timeout
		response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
		response.raise_for_status() # Lança exceção para códigos de erro HTTP (4xx ou 5xx)
		
		result = response.json()
		
		# Extrai a resposta do modelo
		reply = result.get("candidates", [{}])[0]\
					.get("content", {})\
					.get("parts", [{}])[0]\
					.get("text", "")
		
		if not reply:
			feedback = result.get("prompt_feedback", {})
			block_reason = feedback.get("block_reason")
			
			if block_reason:
				reply = f"Desculpe, a resposta foi bloqueada devido ao motivo: {block_reason}."
			else:
				# Pode acontecer se o contexto (planilha) for muito longo
				reply = "Desculpe, não entendi a resposta do modelo (Resposta vazia inesperada). Tente uma pergunta mais específica ou verifique se o seu arquivo não excede o limite de tokens."
		
		return jsonify({"reply": reply})

	except requests.exceptions.Timeout:
		return jsonify({"error": f"Erro de conexão (Timeout): A requisição excedeu o tempo limite ({REQUEST_TIMEOUT}s)."}), 500
	
	except requests.exceptions.RequestException as e:
		print(f"ERRO DE CONEXÃO/API: {e}")
		# Este é o erro 404/Not Found que você está recebendo.
		return jsonify({"error": f"**Erro de conexão (HTTP/Rede):** {str(e)}"}), 500

	except Exception as e:
		print(f"ERRO INESPERADO NO CHAT: {e}")
		return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == "__main__":
	app.run(debug=True)
