import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import chardet
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
import torch

def detect_encoding(file):
    raw_data = file.read()
    file.seek(0)  # Reset file pointer
    result = chardet.detect(raw_data)
    return result['encoding']

def calculate_cluster_coherence(embeddings, cluster_labels):
    unique_labels = np.unique(cluster_labels)
    coherences = []
    for label in unique_labels:
        cluster_embeddings = embeddings[cluster_labels == label]
        if len(cluster_embeddings) > 1:
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            coherence = 1 / (1 + np.mean(distances))
            coherences.append(coherence)
    return coherences

st.title("Semantic Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Community Detection", "Agglomerative", "K-means"],
    help="**Community Detection:** Finds natural groups in your data.\n\nWhen to use: If you're unsure about the number of groups you need.\n\n**Agglomerative:** Groups keywords based on their similarity, step by step.\n\nWhen to use: If you want control over the size of the groups by adjusting the threshold.\n\n**K-means:** Creates a fixed number of groups based on keyword similarity.\n\nWhen to use: If you already know how many groups you want."
)

if clustering_method == "Community Detection":
    cluster_accuracy = st.sidebar.slider("Cluster Accuracy (0-100)", 0, 100, 91) / 100
    min_cluster_size = st.sidebar.number_input("Minimum Cluster Size", min_value=1, max_value=100, value=3)
elif clustering_method == "Agglomerative":
    distance_threshold = st.sidebar.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
elif clustering_method == "K-means":
    n_clusters = st.sidebar.number_input("Number of Clusters for K-means", min_value=2, max_value=50000, value=50)

transformer = st.selectbox(
    "Select Transformer Model",
    ['distiluse-base-multilingual-cased-v2', 'paraphrase-multilingual-mpnet-base-v2', 'all-MiniLM-L6-v2'],
    help="**distiluse-base-multilingual-cased-v2:** Supports 50+ languages, good for multilingual datasets.\n\n**paraphrase-multilingual-mpnet-base-v2:** Very accurate, supports 100+ languages.\n\n**all-MiniLM-L6-v2:** Fast, but primarily for English."
)

uploaded_file = st.file_uploader("Upload Keyword CSV or XLSX", type=["csv", "xlsx"])

if uploaded_file:
    try:
        encoding = detect_encoding(uploaded_file)
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding=encoding)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.write("File loaded successfully!")
        st.write(f"Detected encoding: '{encoding}'")

        # Allow user to select the keyword column
        keyword_column = st.selectbox("Select the column containing keywords", df.columns)

        if keyword_column:
            st.write(f"Total rows: {len(df)}")
            st.write(f"Number of unique keywords: {df[keyword_column].nunique()}")
            
            df[keyword_column] = df[keyword_column].astype(str)
            
            st.write("Sample of the data (first 5 rows):")
            st.write(df[keyword_column].head())

            model = SentenceTransformer(transformer)
            corpus_sentences = df[keyword_column].tolist()
            corpus_embeddings = model.encode(corpus_sentences, batch_size=256, show_progress_bar=True, convert_to_tensor=True)

            if clustering_method == "Community Detection":
                clusters = util.community_detection(corpus_embeddings, min_community_size=min_cluster_size, threshold=cluster_accuracy)
                cluster_labels = np.array([i for i, cluster in enumerate(clusters) for _ in cluster])
            elif clustering_method == "Agglomerative":
                clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
                cluster_labels = clustering_model.fit_predict(corpus_embeddings.cpu().numpy())
            elif clustering_method == "K-means":
                clustering_model = KMeans(n_clusters=n_clusters)
                cluster_labels = clustering_model.fit_predict(corpus_embeddings.cpu().numpy())

            df['Cluster'] = cluster_labels
            df['Cluster'] = df['Cluster'].apply(lambda x: f"Cluster {x + 1}")

            result_df = df.groupby('Cluster')[keyword_column].apply(', '.join).reset_index()
            result_df.columns = ['Cluster', 'Keywords']

            st.write(result_df)

            if clustering_method != "Community Detection":
                cluster_coherences = calculate_cluster_coherence(corpus_embeddings.cpu().numpy(), cluster_labels)
                overall_coherence = np.mean(cluster_coherences)

                st.write(f"Overall Clustering Coherence: {overall_coherence:.4f}")
                st.write("Cluster Coherences:")
                for cluster_id, coherence in enumerate(cluster_coherences):
                    st.write(f"Cluster {cluster_id + 1}: {coherence:.4f}")

            csv_data_clustered = result_df.to_csv(index=False)
            st.download_button(
                label="Download Clustered Keywords",
                data=csv_data_clustered,
                file_name="Clustered_Keywords.csv",
                mime="text/csv"
            )

    except pd.errors.EmptyDataError:
        st.error("EmptyDataError: No columns to parse from file. Please upload a valid CSV or XLSX file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please check your data and try again.")
