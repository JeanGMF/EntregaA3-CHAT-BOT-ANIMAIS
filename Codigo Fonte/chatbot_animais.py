import os
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ARQUIVO_BASE = "base_conhecimento.txt"

# =================================================
# 1. FUNÇÕES AUXILIARES (Limpeza e Arquivo)
# =================================================

def normalizar(texto):
    """Remove acentos e deixa minúsculo para melhorar a comparação."""
    if not isinstance(texto, str): return ""
    texto = texto.lower().strip()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) 
                    if unicodedata.category(c) != 'Mn')
    return texto

def carregar_base():
    """Lê o arquivo e retorna listas prontas para o uso."""
    perguntas_orig = []
    perguntas_norm = []
    respostas = []

    if not os.path.exists(ARQUIVO_BASE):
        # Cria o arquivo se não existir
        with open(ARQUIVO_BASE, "w", encoding="utf-8") as f: pass

    with open(ARQUIVO_BASE, "r", encoding="utf-8") as f:
        for linha in f:
            if "|" not in linha: continue
            partes = linha.split("|", 1) # Divide apenas no primeiro '|'
            if len(partes) < 2: continue
            
            p, r = partes
            perguntas_orig.append(p.strip())
            perguntas_norm.append(normalizar(p)) # Salva versão "limpa"
            respostas.append(r.strip())

    return perguntas_orig, perguntas_norm, respostas

def aprender_nova_pergunta(pergunta_usuario, nova_resposta):
    """Salva a nova interação no final do arquivo."""
    with open(ARQUIVO_BASE, "a", encoding="utf-8") as f:
        # Garante que vai para uma nova linha
        f.write(f"\n{pergunta_usuario}|{nova_resposta}")

# =================================================
# 2. CÉREBRO DO CHATBOT
# =================================================

def treinar(perguntas_norm):
    """Transforma as perguntas em números (Matriz TF-IDF)."""
    if not perguntas_norm: return None, None
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(perguntas_norm)
    return vectorizer, matriz_tfidf

def chatbot():
    # Carrega dados iniciais
    perguntas_orig, perguntas_norm, respostas = carregar_base()
    vectorizer, matriz_tfidf = treinar(perguntas_norm)

    print("="*40)
    print("🦁 Chatbot de Animais Iniciado! 🐼")
    print("Digite 'sair' para encerrar.")
    print("="*40)

    while True:
        usuario = input("\nVocê: ").strip()
        
        if usuario.lower() in ["sair", "tchau", "exit", "fim"]:
            print("Chatbot: Até mais! 🐾")
            break
            
        if not usuario: continue

        # Se a base estiver vazia, pula direto para o aprendizado
        confianca = 0
        indice_melhor = -1
        
        if vectorizer:
            # 1. Normaliza e Vetoriza a pergunta do usuário
            usuario_norm = normalizar(usuario)
            vetor_usuario = vectorizer.transform([usuario_norm])

            # 2. Calcula similaridade
            similaridades = cosine_similarity(vetor_usuario, matriz_tfidf)
            indice_melhor = np.argmax(similaridades)
            confianca = similaridades[0][indice_melhor]

        # 3. Decide se responde ou aprende
        # Limiar de 0.4 (40%) evita respostas erradas
        if confianca > 0.4:
            print(f"Chatbot: {respostas[indice_melhor]}")
            # Debug opcional: mostre a confiança para ajustar se quiser
            # print(f"(Confiança: {confianca:.2f})") 
        else:
            print("Chatbot: Não sei a resposta... 😢")
            nova_resposta = input("Como devo responder isso? (ou Enter para pular): ")
            
            if nova_resposta.strip():
                aprender_nova_pergunta(usuario, nova_resposta)
                print("Chatbot: Aprendido! 🧠")
                
                # Recarrega a base para já usar o conhecimento novo
                perguntas_orig, perguntas_norm, respostas = carregar_base()
                vectorizer, matriz_tfidf = treinar(perguntas_norm)

if __name__ == "__main__":
    chatbot()