import streamlit as st
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load dataset
data_path = "/D A T A' S/arxiv_ml.csv"
df = pd.read_csv(data_path)

# Ambil kolom teks utama untuk digunakan sebagai dataset
if "abstract" in df.columns:
    dataset = df["abstract"].dropna().tolist()
else:
    st.error("Kolom 'abstract' tidak ditemukan dalam dataset.")
    st.stop()

# Inisialisasi TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

# Load Model T5
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Streamlit UI
st.title("QA System dengan RAG (TD-IDF + T5)")
question = st.text_input("Masukkan pertanyaan tentang Machine Learning: ")

if question:
    # TF-IDF Retrieval
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X).flatten()
    best_idx = similarities.argmax()
    retrieved_context = dataset[best_idx]
    
    # Generasi Jawaban dengan T5
    input_text = f"question: {question} context: {retrieved_context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=50)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Tampilkan hasil
    st.write("### Konteks yang ditemukan:")
    st.info(retrieved_context)
    st.write("### Jawaban:")
    st.success(answer)
