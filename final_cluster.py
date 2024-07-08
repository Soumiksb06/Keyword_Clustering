import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
import chardet

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

def preprocess_names(names):
    processed_names = []
    for name in names:
        name = name.lower().replace('dr ', '').replace('gynaecologist', '').strip()
        processed_names.append(name)
    return processed_names

def custom_clustering_rules(names, clusters, exclusion_pairs):
    for pair in exclusion_pairs:
        for cluster in clusters:
            if pair[0] in cluster and pair[1] in cluster:
                clusters.remove(cluster)
                clusters.append([name for name in cluster if name != pair[1]])
                clusters.append([pair[1]])
    return clusters

def detect_encoding(file):
    raw_data = file.read()
    file.seek(0)  # Reset file pointer
    result = chardet.detect(raw_data)
    return result['encoding']

st.title("Semantic Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Agglomerative", "K-means"]
)

if clustering_method == "Agglomerative":
    distance_threshold = st.sidebar.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
elif clustering_method == "K-means":
    n_clusters = st.number_input("Number of Clusters for K-means", min_value=2, max_value=50000, value=50)

transformer = st.selectbox(
    "Select Transformer Model",
    ['all-mpnet-base-v2', 'multi-qa-mpnet-base-cos-v1', 'all-MiniLM-L12-v2']
)

uploaded_file = st.file_uploader("Upload Keyword CSV or XLSX", type=["csv", "xlsx"])

exclusion_pairs = st.text_area("Enter exclusion pairs (one pair per line, names separated by a comma):")

exclusion_list = []
if exclusion_pairs:
    exclusion_lines = exclusion_pairs.split('\n')
    for line in exclusion_lines:
        if ',' in line:
            pair = tuple(map(str.strip, line.split(',')))
            exclusion_list.append(pair)

if uploaded_file:
    try:
        encoding = detect_encoding(uploaded_file)
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding=encoding)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.write("File loaded successfully!")
        st.write(f"Detected encoding: '{encoding}'")

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
            st.write(df['Keyword'].head())

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
                
                if clustering_method == "Agglomerative":
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

                uncluster_percent = (remaining / len(df)) * 100
                clustered_percent = 100 - uncluster_percent
                st.write(f"{clustered_percent:.2f}% of rows clustered successfully!")
                st.write(f"Number of iterations: {iterations}")
                st.write(f"Total unclustered keywords: {remaining}")

                st.write(f"Number of clusters: {len(df['Cluster Name'].unique())}")

                if len(corpus_embeddings) > 1:
                    cluster_coherences = calculate_cluster_coherence(corpus_embeddings.cpu().numpy(), cluster_labels)
                    overall_coherence = np.mean(cluster_coherences)

                    st.write(f"Overall Clustering Coherence: {overall_coherence:.4f}")
                    st.write("Cluster Coherences:")
                    cluster_names = df.groupby('Cluster Name')['Keyword'].first()
                    for (cluster_id, cluster_name), coherence in zip(cluster_names.items(), cluster_coherences):
                        st.write(f"{cluster_name}: {coherence:.4f}")

                # Applying custom clustering rules
                clusters = [[] for _ in range(len(np.unique(cluster_labels)))]
                for name, label in zip(df['Keyword'], df['Cluster Name']):
                    cluster_id = int(label.split(' ')[1]) - 1
                    clusters[cluster_id].append(name)
                
                clusters = custom_clustering_rules(processed_keywords, clusters, exclusion_list)

                result_df = pd.DataFrame([{'Cluster': f'Cluster {i+1}', 'Keywords': ', '.join(cluster)} for i, cluster in enumerate(clusters)])
                
                st.write(result_df)

                csv_data_clustered = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Clustered Keywords",
                    data=csv_data_clustered,
                    file_name="Clustered_Keywords.csv",
                    mime="text/csv"
                )

                if remaining > 0:
                    st.write("Unclustered Keywords:")
                    st.write(list(corpus_set))
                    
                    unclustered_df = pd.DataFrame(list(corpus_set), columns=['Unclustered Keyword'])
                    
                    csv_data_unclustered = unclustered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Unclustered Keywords",
                        data=csv_data_unclustered,
                        file_name="Unclustered_Keywords.csv",
                        mime="text/csv"
                    )
    except Exception as e:
        st.error(f"Error loading the file: {e}")
