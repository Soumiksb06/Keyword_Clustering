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
    for location in specific_locations:
        if re.search(r'\b' + re.escape(location) + r'\b', text, re.IGNORECASE):
            return location

    general_locations = ["chennai", "patna", "kannur", "kerala", "trivandrum",
                         "bangalore", "india", "faridabad", "jaipur", "noida",
                         "meerut", "greater noida", "jammu", "kolkata", "mumbai",
                         "dwarka", "gurugram"]
    for location in general_locations:
        if re.search(r'\b' + re.escape(location) + r'\b', text, re.IGNORECASE):
            return location

    return None


def preprocess_keywords(df):
    df['Location'] = df['Keyword'].apply(extract_location)
    delhi_keywords = df[df['Location'] == 'delhi']
    gurgaon_keywords = df[df['Location'] == 'gurgaon']
    pune_keywords = df[df['Location'] == 'pune']
    other_keywords = df[~df['Location'].isin(['delhi', 'gurgaon', 'pune'])]
    return delhi_keywords, gurgaon_keywords, pune_keywords, other_keywords


def cluster_keywords(keywords, model, clustering_method, n_clusters=None, distance_threshold=None, cluster_accuracy=None, min_cluster_size=None):
    embeddings = model.encode(keywords['Keyword'].tolist(), convert_to_tensor=True)
    
    if clustering_method == "K-means":
        clustering_model = KMeans(n_clusters=n_clusters)
        cluster_labels = clustering_model.fit_predict(embeddings)
    elif clustering_method == "Agglomerative":
        clustering_model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
        cluster_labels = clustering_model.fit_predict(embeddings)
    else:
        # Implement Community Detection logic here if needed
        pass

    keywords['Cluster'] = cluster_labels
    return keywords


st.title("Semantic Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Agglomerative", "K-means"],
    help="**Agglomerative:** Groups keywords based on their similarity, step by step.\n\nWhen to use: If you want control over the size of the groups by adjusting the threshold.\n\n**K-means:** Creates a fixed number of groups based on keyword similarity.\n\nWhen to use: If you already know how many groups you want."
)

if clustering_method == "Agglomerative":
    distance_threshold = st.sidebar.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
elif clustering_method == "K-means":
    n_clusters = st.number_input("Number of Clusters for K-means", min_value=2, max_value=50000, value=50)

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

        delhi_keywords, gurgaon_keywords, pune_keywords, other_keywords = preprocess_keywords(df)

        model = SentenceTransformer(transformer)

        if not delhi_keywords.empty:
            delhi_keywords = cluster_keywords(delhi_keywords, model, clustering_method, n_clusters, distance_threshold)
        if not gurgaon_keywords.empty:
            gurgaon_keywords = cluster_keywords(gurgaon_keywords, model, clustering_method, n_clusters, distance_threshold)
        if not pune_keywords.empty:
            pune_keywords = cluster_keywords(pune_keywords, model, clustering_method, n_clusters, distance_threshold)
        if not other_keywords.empty:
            other_keywords = cluster_keywords(other_keywords, model, clustering_method, n_clusters, distance_threshold)

        clustered_df = pd.concat([delhi_keywords, gurgaon_keywords, pune_keywords, other_keywords])
        st.write(clustered_df)

        # Visualize clusters
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(model.encode(clustered_df['Keyword'].tolist(), convert_to_tensor=True))
        
        fig = go.Figure()

        unique_labels = clustered_df['Cluster'].unique()
        colors = generate_colors(len(unique_labels))

        for label, color in zip(unique_labels, colors):
            cluster_points = embeddings_2d[clustered_df['Cluster'] == label]
            fig.add_trace(go.Scatter(x=cluster_points[:, 0], y=cluster_points[:, 1],
                                     mode='markers', marker=dict(color='rgba' + str(color + (0.8,)), size=8),
                                     name=f'Cluster {label}'))

        fig.update_layout(title="Keyword Clusters",
                          xaxis_title="PCA Component 1",
                          yaxis_title="PCA Component 2",
                          showlegend=True)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
