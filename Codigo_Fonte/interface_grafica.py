import tkinter as tk
from tkinter import scrolledtext
import chatbot_animais as cerebro  # Seu cérebro original
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading

# --- Configuração Inicial ---
# Carrega o cérebro apenas uma vez
print("Carregando IA...")
perguntas_orig, perguntas_norm, respostas = cerebro.carregar_base()
vectorizer, matriz_tfidf = cerebro.treinar(perguntas_norm)
print("IA Carregada!")

def processar_resposta():
    # 1. Pega o texto do usuário
    pergunta_usuario = entrada_texto.get().strip()
    if not pergunta_usuario: return

    # 2. Mostra no chat (Visual)
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, f"Você: {pergunta_usuario}\n", "user")
    entrada_texto.delete(0, tk.END) # Limpa o campo

    # 3. Inteligência Artificial (Lógica)
    resposta_final = "Desculpe, não entendi."
    
    if vectorizer:
        usuario_norm = cerebro.normalizar(pergunta_usuario)
        vetor_usuario = vectorizer.transform([usuario_norm])
        similaridades = cosine_similarity(vetor_usuario, matriz_tfidf)
        indice_melhor = np.argmax(similaridades)
        confianca = similaridades[0][indice_melhor]

        if confianca > 0.4:
            resposta_final = respostas[indice_melhor]
        else:
            resposta_final = "Não sei a resposta (Modo aprendizado apenas no terminal). 🐢"

    # 4. Mostra resposta do Bot
    chat_area.insert(tk.END, f"Bot: {resposta_final}\n\n", "bot")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END) # Rola para baixo

# --- Criação da Janela (Visual) ---
janela = tk.Tk()
janela.title("🦁 ZooBot - Projeto A3")
janela.geometry("500x600")
janela.configure(bg="#f0f0f0")

# Cabeçalho
titulo = tk.Label(janela, text="🦁 Chatbot de Zoologia", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
titulo.pack(pady=10)

# Área de Chat (Rolagem)
chat_area = scrolledtext.ScrolledText(janela, wrap=tk.WORD, width=50, height=20, font=("Arial", 10))
chat_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
chat_area.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
chat_area.tag_config("bot", foreground="green")
chat_area.insert(tk.END, "Bot: Olá! Pergunte sobre animais. 🐾\n\n", "bot")
chat_area.config(state=tk.DISABLED) # Impede digitar direto na área de texto

# Área de Entrada
frame_entrada = tk.Frame(janela, bg="#f0f0f0")
frame_entrada.pack(padx=10, pady=10, fill=tk.X)

entrada_texto = tk.Entry(frame_entrada, font=("Arial", 12))
entrada_texto.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
entrada_texto.bind("<Return>", lambda event: processar_resposta()) # Envia com Enter

botao_enviar = tk.Button(frame_entrada, text="Enviar ➤", command=processar_resposta, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
botao_enviar.pack(side=tk.RIGHT)

# Inicia o programa
janela.mainloop()