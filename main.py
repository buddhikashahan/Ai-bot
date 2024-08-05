import fitz  
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\d', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text.lower()


def split_text_into_chunks(text, chunk_size=100):  
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def get_embeddings(text_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list, convert_to_tensor=True)
    return embeddings


def get_answer(question, document_chunks, document_embeddings, top_k=3):
    question_embedding = get_embeddings([question])[0]
    similarities = cosine_similarity(question_embedding.reshape(1, -1), document_embeddings)
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]  
    answers = [document_chunks[i] for i in top_indices]
    return "\n\n".join(answers)


def chat_system(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    clean_text = preprocess_text(pdf_text)
    document_chunks = split_text_into_chunks(clean_text)
    document_embeddings = get_embeddings(document_chunks)

    print("Chatbot is ready. You can start asking questions or type 'exit' to end the session.")

    while True:
        question = input("You: ")
        if question.lower() in ['exit', 'quit']:
            break

        answer = get_answer(question, document_chunks, document_embeddings)
        print(f"Bot: {answer}")


if __name__ == "__main__":
    pdf_path = 'D:/history.txt'  
    chat_system(pdf_path)

