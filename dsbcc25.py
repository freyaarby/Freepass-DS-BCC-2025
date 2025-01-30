import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer

#Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Load dataset
DATA_PATH = "arvix_ml.csv"  
try:
    df = pd.read_csv(DATA_PATH)
    if "abstract" not in df.columns:
        st.error("Kolom 'abstract' tidak ditemukan dalam dataset.")
        st.stop()
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

#Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

#Apply preprocessing
df["processed_abstract"] = df["abstract"].astype(str).apply(preprocess_text)

#TF-IDF setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["processed_abstract"])

def retrieve_documents(query, top_k=5):
    query_vec = vectorizer.transform([preprocess_text(query)])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][["title", "abstract"]]

#Load T5 model
@st.cache_resource()
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_answer(question, context):
    input_text = f"question: {question}  context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

#UI Streamlit
st.title("QA Machine Learning")
query = st.text_input("Masukkan pertanyaan Anda:")

if query:
    retrieved_docs = retrieve_documents(query)
    if not retrieved_docs.empty:
        st.subheader("Dokumen terkait:")
        for _, row in retrieved_docs.iterrows():
            st.write(f"**{row['title']}**")
            st.write(row['abstract'])
            st.write("---")
        
        context = retrieved_docs.iloc[0]["abstract"]
        answer = generate_answer(query, context)
        st.subheader("Jawaban yang dihasilkan:")
        st.write(answer)
    else:
        st.write("Tidak ada dokumen yang ditemukan.")