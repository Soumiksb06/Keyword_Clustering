import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cluster_coherence(clusters):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_coherences = {}
    all_embeddings = []
    all_keywords = []

    for cluster_label, keywords in clusters.items():
        embeddings = model.encode(keywords)
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        cluster_coherence = similarity_matrix.mean()
        cluster_coherences[cluster_label] = cluster_coherence
        all_embeddings.extend(embeddings)
        all_keywords.extend(keywords)

    overall_coherence = np.mean(list(cluster_coherences.values()))
    all_embeddings = np.array(all_embeddings)
    inter_cluster_similarity = cosine_similarity(all_embeddings).mean()

    return cluster_coherences, overall_coherence, inter_cluster_similarity

def perform_spectral_clustering(phrases, model_name, n_clusters):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(phrases)

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    cluster_labels = spectral.fit_predict(embeddings)

    clusters = {}
    noise_points = []
    for i, label in enumerate(cluster_labels):
        if label == -1:
            noise_points.append(phrases[i])
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(phrases[i])

    cluster_coherences, overall_coherence, inter_cluster_similarity = calculate_cluster_coherence(clusters)
    return clusters, noise_points, overall_coherence, cluster_coherences, inter_cluster_similarity

def perform_agglomerative_clustering(phrases, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(phrases)

    agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage='ward')
    cluster_labels = agglo.fit_predict(embeddings)

    clusters = {}
    noise_points = []
    for i, label in enumerate(cluster_labels):
        if label == -1:
            noise_points.append(phrases[i])
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(phrases[i])

    cluster_coherences, overall_coherence, inter_cluster_similarity = calculate_cluster_coherence(clusters)
    return clusters, noise_points, overall_coherence, cluster_coherences, inter_cluster_similarity

def get_semantic_name(keywords, model):
    embeddings = model.encode(keywords)
    similarity_matrix = cosine_similarity(embeddings)
    avg_similarities = similarity_matrix.mean(axis=1)
    semantic_name = keywords[np.argmax(avg_similarities)]
    return semantic_name

st.title("Keyword Clustering App")

uploaded_file = st.file_uploader("Upload a CSV file with a column named 'Keyword'", type="csv")

if uploaded_file:
    phrases = pd.read_csv(uploaded_file)['Keyword'].tolist()
    total_phrases = len(phrases)

    model_name = 'sentence-transformers/all-mpnet-base-v2'

    st.sidebar.subheader("Select Clustering Algorithm")
    clustering_algorithm = st.sidebar.radio("Algorithm", ("Spectral Clustering", "Agglomerative Clustering"))

    if clustering_algorithm == "Spectral Clustering":
        n_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=20, value=5)
        clusters, noise_points, overall_coherence, cluster_coherences, inter_cluster_similarity = perform_spectral_clustering(phrases, model_name, n_clusters)
    else:
        clusters, noise_points, overall_coherence, cluster_coherences, inter_cluster_similarity = perform_agglomerative_clustering(phrases, model_name)

    final_clusters = {}
    final_noise_points = []
    final_coherences = []

    for label, coherence in cluster_coherences.items():
        if coherence < 0.55:
            final_noise_points.extend(clusters[label])
        else:
            final_clusters[label] = clusters[label]
            final_coherences.append(coherence)

    if final_coherences:
        final_overall_coherence = np.mean(final_coherences)
    else:
        final_overall_coherence = 0

    model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_clusters = {}
    used_names = set()
    for label, keywords in final_clusters.items():
        semantic_name = get_semantic_name(keywords, model)
        if semantic_name in used_names:
            count = 1
            new_semantic_name = f"{semantic_name}_{count}"
            while new_semantic_name in used_names:
                count += 1
                new_semantic_name = f"{semantic_name}_{count}"
            semantic_name = new_semantic_name
        used_names.add(semantic_name)
        semantic_clusters[semantic_name] = keywords

    st.subheader("Final Clusters:")
    for semantic_name, keywords in semantic_clusters.items():
        st.markdown(f"**{semantic_name}**:")
        for phrase in keywords:
            st.markdown(f"- {phrase}")

    if final_noise_points:
        st.subheader("Final Noise points (unclustered):")
        for phrase in final_noise_points:
            st.markdown(f"- {phrase}")

    st.write(f"Final noise ratio: {len(final_noise_points) / total_phrases:.2%}")
    st.write(f"Final overall coherence: {final_overall_coherence:.4f}")

    ranked_clusters = sorted([(label, coherence) for label, coherence in cluster_coherences.items() if coherence >= 0.55], key=lambda x: x[1], reverse=True)
    st.subheader("Clusters ranked by coherence:")
    for rank, (label, coherence) in enumerate(ranked_clusters, 1):
        semantic_name = get_semantic_name(final_clusters[label], model)
        st.markdown(f"{rank}. **{semantic_name}**: {coherence:.4f}")

    save_path = 'final_clusters.csv'
    cluster_data = []

    for label, keywords in final_clusters.items():
        semantic_name = get_semantic_name(keywords, model)
        cluster_data.append({'Cluster': semantic_name, 'Keywords': ', '.join(keywords)})

    if final_noise_points:
        cluster_data.append({'Cluster': 'Unclustered', 'Keywords': ', '.join(final_noise_points)})

    df = pd.DataFrame(cluster_data)
    df.to_csv(save_path, index=False)

    st.success(f"Cluster Data saved to CSV file at {save_path}.")
