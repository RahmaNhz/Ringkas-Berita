import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Download NLTK package punkt
nltk.download('punkt')
nltk.download("stopwords")

# Fungsi untuk memproses dan meringkas teks
def summarize_text(text, num_sentences=3):
    # Tokenisasi kalimat
    kalimat = nltk.sent_tokenize(text)

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
    # Memotong kalimat untuk label graf
    def potong_kalimat(kalimat, max_len=40):
        if len(kalimat) > max_len:
            return kalimat[:max_len] + '...'
        return kalimat

    labels = {i: potong_kalimat(kalimat[i]) for i in range(len(kalimat))}

    # Plot graf
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
#num_sentences = st.slider("Pilih Jumlah Kalimat Ringkasan", min_value=2, max_value=5, value=3)
num_sentences = st.selectbox(
    "Pilih Jumlah Kalimat Ringkasan", 
    options=[2, 3, 4, 5], 
    index=2  # Set default selection to 3 (index 2)
)

if st.button("Buat Ringkasan"):
    if text:
        # Dapatkan ringkasan, graf, dan cosine similarity
        summary, G, kalimat, cosine = summarize_text(text, num_sentences=num_sentences)

        # Tampilkan ringkasan
        st.subheader("Ringkasan Berita:")
        st.write(summary)

        # Tampilkan graf antar kalimat
        st.subheader("Graf Antar Kalimat Berdasarkan Cosine Similarity:")
        plot_graph(G, kalimat)
       # Menampilkan nilai betweenness centrality tertinggi untuk setiap kalimat
        st.subheader("Centrality Tertinggi untuk Setiap Kalimat:")

        # Ambil betweenness centrality dari graf
        betweenness_centrality = nx.betweenness_centrality(G)

        # List untuk menyimpan centrality dan indeks kalimat
        centrality_scores = [(i, betweenness_centrality[i]) for i in range(len(kalimat))]

        # Urutkan berdasarkan nilai betweenness centrality tertinggi
        sorted_centrality_scores = sorted(centrality_scores, key=lambda x: x[1], reverse=True)

        # Tampilkan hasil yang sudah diurutkan
        for index, centrality in sorted_centrality_scores:
            st.write(f"Kalimat {index + 1}: {kalimat[index]}")
            st.write(f"Betweenness Centrality: {centrality:.4f}")
            st.write("---")

    else:
        st.warning("Harap masukkan teks berita terlebih dahulu.")
