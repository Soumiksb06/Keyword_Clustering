import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import chardet
from detect_delimiter import detect
import numpy as np
import plotly.graph_objects as go
import colorsys
from sklearn.cluster import AgglomerativeClustering, KMeans
import torch
from sklearn.decomposition import PCA

def generate_colors(num_colors):
    colors = []
    hue_values = np.linspace(0, 1, num_colors, endpoint=False)
    np.random.shuffle(hue_values)
    
    for hue in hue_values:
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

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
st.write("**Note:** The dataset must contain a column named 'Keyword', 'keywords', 'query', 'Top queries', 'queries', or 'Search terms report'.")

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Community Detection", "Agglomerative", "K-means"],
    help="Community Detection: Best for discovering organic clusters with varying sizes.\n\nAgglomerative: Useful for hierarchical clustering with a defined distance threshold.\n\nK-means: Effective when you have a predefined number of clusters."
)

if clustering_method == "Community Detection":
    cluster_accuracy = st.slider("Cluster Accuracy (0-100)", 0, 100, 80) / 100
    min_cluster_size = st.number_input("Minimum Cluster Size", min_value=1, max_value=100, value=3)
    st.write("**Tip:** Increase the cluster accuracy if clusters seem too large or unrelated keywords are grouped together. Decrease it if clusters are too small or numerous.")
elif clustering_method == "Agglomerative":
    distance_threshold = st.sidebar.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
    st.write("**Tip:** Increase the distance threshold to form fewer, larger clusters. Decrease it to form more, smaller clusters.")
elif clustering_method == "K-means":
    n_clusters = st.number_input("Number of Clusters for K-means", min_value=2, max_value=100, value=5)

transformer = st.selectbox(
    "Select Transformer Model",
    ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-mpnet-base-v2'],
    help="all-MiniLM-L6-v2: Lightweight and fast, suitable for small datasets.\n\nall-mpnet-base-v2: Balanced between performance and speed, good for medium-sized datasets.\n\nparaphrase-mpnet-base-v2: High accuracy, ideal for large datasets and detailed analysis."
)

uploaded_file = st.file_uploader("Upload Keyword CSV or XLSX", type=["csv", "xlsx"])

if uploaded_file:
    try:
        raw_data = uploaded_file.getvalue()
        detected = chardet.detect(raw_data)
        encoding_type = detected['encoding'] if detected['confidence'] >= 0.8 else 'utf-8'
        
        try:
            text_data = raw_data.decode(encoding_type)
        except UnicodeDecodeError:
            encoding_type = 'ISO-8859-1'
            text_data = raw_data.decode(encoding_type)

        firstline = text_data.splitlines()[0]
        
        if uploaded_file.name.endswith('.csv'):
            delimiter_type = detect(firstline)
            df = pd.read_csv(uploaded_file, encoding=encoding_type, delimiter=delimiter_type or ',')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.write("File loaded successfully!")
        st.write(f"Detected encoding: '{encoding_type}'")

        df.rename(columns={"Search term": "Keyword", "keyword": "Keyword", "query": "Keyword", "Top queries": "Keyword", "queries": "Keyword", "Keywords": "Keyword", "keywords": "Keyword", "Search terms report": "Keyword"}, inplace=True)

        if "Keyword" not in df.columns:
            st.error("Error! Please make sure your CSV or XLSX file contains a column named 'Keyword'!")
        else:
            st.write(f"Total rows: {len(df)}")
            st.write(f"Number of unique keywords: {df['Keyword'].nunique()}")
            st.write("Data types in 'Keyword' column:")
            st.write(df['Keyword'].apply(type).value_counts())
            
            df['Keyword'] = df['Keyword'].astype(str)
            
            st.write("Sample of the data (first 5 rows):")
            st.write(df.head())

            model = SentenceTransformer(transformer)
            corpus_set = set(df['Keyword'])
            corpus_set_all = corpus_set
            cluster_name_list = []
            corpus_sentences_list = []
            df_all = []
            cluster = True
            iterations = 0

            while cluster:
                corpus_sentences = list(corpus_set)
                check_len = len(corpus_sentences)

                if len(corpus_sentences) == 0:
                    break

                corpus_embeddings = model.encode(corpus_sentences, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
                
                if len(corpus_embeddings.shape) == 1:
                    corpus_embeddings = corpus_embeddings.reshape(1, -1)
                
                if clustering_method == "Community Detection":
                    clusters = util.community_detection(corpus_embeddings, min_community_size=min_cluster_size, threshold=cluster_accuracy)
                    cluster_labels = np.array([i for i, cluster in enumerate(clusters) for _ in cluster])
                    
                    coherences = []
                    for cluster in clusters:
                        if len(cluster) > 1:
                            cluster_embeddings = corpus_embeddings[cluster]
                            centroid = torch.mean(cluster_embeddings, dim=0)
                            distances = torch.norm(cluster_embeddings - centroid, dim=1)
                            coherence = 1 / (1 + torch.mean(distances).item())
                            coherences.append(coherence)
                    
                    if coherences:
                        overall_coherence = np.mean(coherences)
                        st.write(f"Overall Clustering Coherence: {overall_coherence:.4f}")
                        st.write("Cluster Coherences:")
                        for i, coherence in enumerate(coherences):
                            st.write(f"Cluster {i+1}: {coherence:.4f}")
                    else:
                        st.write("Coherence: Not applicable (insufficient data)")
                    
                elif clustering_method == "Agglomerative":
                    max_clusters = len(corpus_sentences) // 4
                    while True:
                        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
                        cluster_labels = clustering_model.fit_predict(corpus_embeddings.cpu().numpy())
                        n_clusters = len(np.unique(cluster_labels))
                        if n_clusters <= max_clusters:
                            break
                        distance_threshold += 0.1
                    st.write(f"Adjusted distance threshold: {distance_threshold:.2f}")
                    st.write(f"Number of clusters formed: {n_clusters}")
                elif clustering_method == "K-means":
                    clustering_model = KMeans(n_clusters=n_clusters)
                    cluster_labels = clustering_model.fit_predict(corpus_embeddings.cpu().numpy())

                if clustering_method == "Community Detection":
                    for keyword, cluster in enumerate(clusters):
                        for sentence_id in cluster:
                            corpus_sentences_list.append(corpus_sentences[sentence_id])
                            cluster_name_list.append(f"Cluster {keyword + 1}, #{len(cluster)} Elements")
                else:
                    for sentence_id, cluster_id in enumerate(cluster_labels):
                        corpus_sentences_list.append(corpus_sentences[sentence_id])
                        cluster_name_list.append(f"Cluster {cluster_id + 1}")

                df_new = pd.DataFrame({'Cluster Name': cluster_name_list, 'Keyword': corpus_sentences_list})

                df_all.append(df_new)
                have = set(df_new["Keyword"])

                corpus_set = corpus_set_all - have
                remaining = len(corpus_set)
                iterations += 1
                if check_len == remaining:
                    break

            if len(df_all) == 0:
                st.error("No clusters were formed. Please check your data or adjust the clustering parameters.")
            else:
                df_new = pd.concat(df_all)
                df = df.merge(df_new.drop_duplicates('Keyword'), how='left', on="Keyword")

                df['Cluster Name'] = df['Cluster Name'].fillna("no_cluster")

                df['Length'] = df['Keyword'].astype(str).map(len)
                df = df.sort_values(by="Length", ascending=True)
                df['Cluster Name'] = df.groupby('Cluster Name')['Keyword'].transform('first')
                df.sort_values(['Cluster Name', "Keyword"], ascending=[True, True], inplace=True)
                df = df.drop('Length', axis=1)

                df = df[['Cluster Name', 'Keyword']]

                df.sort_values(["Cluster Name", "Keyword"], ascending=[True, True], inplace=True)

                st.write("Clustering completed successfully!")
                st.write("Sample of the clustered data (first 10 rows):")
                st.write(df.head(10))

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Clustered Keywords CSV",
                    data=csv,
                    file_name='clustered_keywords.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
