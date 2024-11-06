import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Fungsi manual untuk memisahkan kalimat berdasarkan tanda titik
def sentence_tokenize(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence]

# Fungsi untuk memproses dan meringkas teks
def summarize_text(text, num_sentences=3):
    # Tokenisasi kalimat tanpa NLTK
    kalimat = sentence_tokenize(text)

    # Menghitung TF-IDF untuk setiap kalimat
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(kalimat)

    # Menghitung cosine similarity
    cosine = cosine_similarity(tfidf_matrix)
    
    # Membuat graf berdasarkan cosine similarity
    G = nx.Graph()
    for i in range(len(cosine)):
        G.add_node(i)
    for i in range(len(cosine)):
        for j in range(len(cosine)):
            similarity = cosine[i][j]
            if similarity >= 0.05 and i != j:
                G.add_edge(i, j)
    
    # Menghitung betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Urutkan kalimat berdasarkan centrality tertinggi
    ranked_sentences = sorted(((betweenness_centrality[i], s) for i, s in enumerate(kalimat)), reverse=True)
    
    # Ambil kalimat untuk ringkasan
    summary_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]
    summary = ' '.join(summary_sentences)
    
    return summary, G, kalimat, cosine

# Fungsi untuk menampilkan graf
def plot_graph(G, kalimat):
    def potong_kalimat(kalimat, max_len=40):
        if len(kalimat) > max_len:
            return kalimat[:max_len] + '...'
        return kalimat

    labels = {i: potong_kalimat(kalimat[i]) for i in range(len(kalimat))}

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=900, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color="black")
    plt.title("Graph berdasarkan Cosine Similarity antar Kalimat")
    st.pyplot(plt)

# Streamlit UI
st.title("Ringkasan Berita dengan Visualisasi Graf")
st.write("Masukkan paragraf berita yang ingin diringkas, lalu pilih jumlah kalimat untuk ringkasan.")

# Input teks
text = st.text_area("Masukkan Teks Berita", height=200)

# Pilihan jumlah kalimat ringkasan
num_sentences = st.selectbox(
    "Pilih Jumlah Kalimat Ringkasan", 
    options=[2, 3, 4, 5], 
    index=2  
)

if st.button("Buat Ringkasan"):
    if text:
        summary, G, kalimat, cosine = summarize_text(text, num_sentences=num_sentences)

        st.subheader("Ringkasan Berita:")
        st.write(summary)

        st.subheader("Graf Antar Kalimat Berdasarkan Cosine Similarity:")
        plot_graph(G, kalimat)

        st.subheader("Centrality Tertinggi untuk Setiap Kalimat:")
        betweenness_centrality = nx.betweenness_centrality(G)
        centrality_scores = [(i, betweenness_centrality[i]) for i in range(len(kalimat))]
        sorted_centrality_scores = sorted(centrality_scores, key=lambda x: x[1], reverse=True)
        
        for index, centrality in sorted_centrality_scores:
            st.write(f"Kalimat {index + 1}: {kalimat[index]}")
            st.write(f"Betweenness Centrality: {centrality:.4f}")
            st.write("---")

    else:
        st.warning("Harap masukkan teks berita terlebih dahulu.")
