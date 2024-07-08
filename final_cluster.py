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
from langdetect import detect_langs
from iso639 import languages
import re


def generate_colors(num_colors):
    colors = []
    hue_values = np.linspace(0, 1, num_colors, endpoint=False)
    np.random.shuffle(hue_values)
    
    for hue in hue_values, lightness, saturation in np.ndindex((50, 60), (90, 100)):
        rgb = colorsys.hls_to_rgb(hue / 100, lightness / 100, saturation / 100)
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


def detect_encoding(file):
    raw_data = file.read()
    file.seek(0)  # Reset file pointer
    result = chardet.detect(raw_data)
    return result['encoding']


def detect_language(text):
    try:
        detected_langs = detect_langs(text)
        main_language_code = detected_langs[0].lang
        language_name = languages.get(part1=main_language_code).name
        return language_name
    except:
        return 'Unknown Language'


def extract_location(text):
    specific_locations = ["delhi", "gurgaon", "pune"]
    other_locations = ["chennai", "patna", "kannur", "kerala", "trivandrum",
                       "bangalore", "india", "faridabad",
                       "jaipur", "noida", "meerut", "greater noida", "jammu",
                       "kolkata", "mumbai", "dwarka", "gurugram"]
    for location in specific_locations + other_locations:
        if re.search(r'\b' + re.escape(location) + r'\b', text, re.IGNORECASE):
            return location
    return None


def cluster_keywords(keywords, method, model_name, additional_params):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(keywords, convert_to_tensor=True)

    if method == "Community Detection":
        import networkx as nx
        from community import community_louvain

        # Calculate cosine similarities and create a graph
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
        edges = [(i, j, float(cosine_scores[i][j])) for i in range(len(keywords)) for j in range(i + 1, len(keywords))]
        G = nx.Graph()
        G.add_weighted_edges_from(edges)

        # Apply community detection algorithm
        partition = community_louvain.best_partition(G, resolution=additional_params['resolution'])
        clusters = [partition[i] for i in range(len(keywords))]

    elif method == "Agglomerative":
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=additional_params['distance_threshold'])
        clusters = clustering_model.fit_predict(embeddings.cpu())

    elif method == "K-means":
        clustering_model = KMeans(n_clusters=additional_params['n_clusters'], random_state=0)
        clusters = clustering_model.fit_predict(embeddings.cpu())

    return clusters, embeddings


st.title("Semantic Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Community Detection", "Agglomerative", "K-means"],
    help="**Community Detection:** Finds natural groups in your data.\n\nWhen to use: If you're unsure about the number of groups you need.\n\n**Agglomerative:** Groups keywords based on their similarity, step by step.\n\nWhen to use: If you want control over the size of the groups by adjusting the threshold.\n\n**K-means:** Creates a fixed number of groups based on keyword similarity.\n\nWhen to use: If you already know how many groups you want."
)

if clustering_method == "Community Detection":
    cluster_accuracy = st.slider("Cluster Accuracy (0-100)", 0, 100, 91) / 100
    min_cluster_size = st.number_input("Minimum Cluster Size", min_value=1, max_value=100, value=3)
    additional_params = {'resolution': cluster_accuracy}
elif clustering_method == "Agglomerative":
    distance_threshold = st.sidebar.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    additional_params = {'distance_threshold': distance_threshold}
elif clustering_method == "K-means":
    n_clusters = st.number_input("Number of Clusters for K-means", min_value=2, max_value=50000, value=50)
    additional_params = {'n_clusters': n_clusters}

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

        df.rename(columns={"Search term": "Keyword", "keyword": "Keyword", "query": "Keyword", "Top queries": "Keyword", "queries": "Keyword"}, inplace=True)

        keywords = df['Keyword'].tolist()

        # Process specific locations (Delhi, Gurgaon, Pune) separately
        specific_keywords = [kw for kw in keywords if extract_location(kw) in ["delhi", "gurgaon", "pune"]]
        other_keywords = [kw for kw in keywords if extract_location(kw) not in ["delhi", "gurgaon", "pune"]]

        # Cluster specific location keywords
        if specific_keywords:
            specific_clusters, specific_embeddings = cluster_keywords(specific_keywords, clustering_method, transformer, additional_params)

        # Cluster other keywords
        if other_keywords:
            other_clusters, other_embeddings = cluster_keywords(other_keywords, clustering_method, transformer, additional_params)

        # Combine results
        combined_keywords = specific_keywords + other_keywords
        combined_clusters = np.concatenate((specific_clusters, other_clusters))

        # Calculate cluster coherence
        combined_embeddings = torch.cat((specific_embeddings, other_embeddings), dim=0)
        coherences = calculate_cluster_coherence(combined_embeddings, combined_clusters)

        st.write(f"Number of clusters: {len(set(combined_clusters))}")
        st.write(f"Average cluster coherence: {np.mean(coherences):.4f}")

        # Create a dataframe with the results
        result_df = pd.DataFrame({'Keyword': combined_keywords, 'Cluster': combined_clusters})

        # Visualize the clusters
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(combined_embeddings.cpu().numpy())
        result_df['x'] = reduced_embeddings[:, 0]
        result_df['y'] = reduced_embeddings[:, 1]

        fig = go.Figure()
        colors = generate_colors(len(set(combined_clusters)))
        for cluster_id in set(combined_clusters):
            cluster_data = result_df[result_df['Cluster'] == cluster_id]
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                marker=dict(color='rgba' + str(colors[cluster_id]), size=10, line=dict(width=2, color='DarkSlateGrey')),
                text=cluster_data['Keyword'],
                name=f'Cluster {cluster_id}'
            ))

        fig.update_layout(title='Keyword Clusters', xaxis_title='PCA Component 1', yaxis_title='PCA Component 2')
        st.plotly_chart(fig)

        # Option to download the clustered keywords
        st.download_button(
            label="Download Clustered Keywords",
            data=result_df.to_csv(index=False),
            file_name='clustered_keywords.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error: {e}")
