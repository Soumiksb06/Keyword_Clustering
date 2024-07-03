import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import hdbscan
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

def perform_clustering(phrases, model_name, min_cluster_size):
    if len(phrases) < 2:
        return {}, phrases, 0, {}, 0

    model = SentenceTransformer(model_name)
    embeddings = model.encode(phrases)
    similarities = cosine_similarity(embeddings)
    distances = 1 - similarities
    distances = np.asarray(distances, dtype=np.float64)
    np.fill_diagonal(distances, 0)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(distances)

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

    models = [
        'sentence-transformers/paraphrase-mpnet-base-v2',
        'sentence-transformers/all-mpnet-base-v2'
    ]

    iteration = 0
    all_clusters = {}
    all_cluster_coherences = {}
    remaining_phrases = phrases.copy()
    noise_ratio_history = []

    while True:
        model = models[iteration % len(models)]
        min_cluster_size = max(5 - iteration, 2)

        clusters, noise_points, overall_coherence, cluster_coherences, inter_cluster_similarity = perform_clustering(remaining_phrases, model, min_cluster_size)

        for label, cluster in clusters.items():
            new_label = len(all_clusters)
            all_clusters[new_label] = cluster
            all_cluster_coherences[new_label] = cluster_coherences[label]

        noise_ratio = len(noise_points) / total_phrases
        noise_ratio_history.append(noise_ratio)

        if overall_coherence > 0.55:
            break

        remaining_phrases = noise_points
        iteration += 1

        if len(remaining_phrases) <= 1 or len(remaining_phrases) == len(phrases):
            break

    final_clusters = {}
    final_noise_points = []
    final_coherences = []

    for label, coherence in all_cluster_coherences.items():
        if coherence < 0.55:
            final_noise_points.extend(all_clusters[label])
        else:
            final_clusters[label] = all_clusters[label]
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
        st.markdown(f"**{semantic_name}** (Coherence: {all_cluster_coherences[semantic_name]:.4f}):")
        for phrase in keywords:
            st.markdown(f"- {phrase}")

    st.subheader("Final Noise points (unclustered):")
    for phrase in final_noise_points:
        st.markdown(f"- {phrase}")

    st.write(f"Final noise ratio: {len(final_noise_points) / total_phrases:.2%}")
    st.write(f"Final overall coherence: {final_overall_coherence:.4f}")

    ranked_clusters = sorted([(label, coherence) for label, coherence in all_cluster_coherences.items() if coherence >= 0.55], key=lambda x: x[1], reverse=True)
    st.subheader("Clusters ranked by coherence:")
    for rank, (label, coherence) in enumerate(ranked_clusters, 1):
        semantic_name = get_semantic_name(final_clusters[label], model)
        st.markdown(f"{rank}. **{semantic_name}**: {coherence:.4f}")

    save_path = '/content/drive/MyDrive/final_clusters.csv'

    cluster_data = []

    for label, keywords in final_clusters.items():
        semantic_name = get_semantic_name(keywords, model)
        cluster_data.append({'Cluster': semantic_name, 'Keywords': ', '.join(keywords)})

    if final_noise_points:
        cluster_data.append({'Cluster': 'Unclustered', 'Keywords': ', '.join(final_noise_points)})

    df = pd.DataFrame(cluster_data)
    df.to_csv(save_path, index=False)

    st.success(f"Cluster Data saved to CSV file at {save_path}.")
