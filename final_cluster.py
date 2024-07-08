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
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

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

def geocode_keyword(geolocator, keyword):
    try:
        location = geolocator.geocode(keyword)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except (GeocoderTimedOut, GeocoderUnavailable):
        time.sleep(1)
        return geocode_keyword(geolocator, keyword)

st.title("Semantic Keyword Clustering Tool with Geography Awareness")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Community Detection", "Agglomerative", "K-means"],
    help="**Community Detection:** Finds natural groups in your data.\n\n**Agglomerative:** Groups keywords based on their similarity, step by step.\n\n**K-means:** Creates a fixed number of groups based on keyword similarity."
)

if clustering_method == "Community Detection":
    cluster_accuracy = st.slider("Cluster Accuracy (0-100)", 0, 100, 91) / 100
    min_cluster_size = st.number_input("Minimum Cluster Size", min_value=1, max_value=100, value=3)
elif clustering_method == "Agglomerative":
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

        df.rename(columns={"Search term": "Keyword", "keyword": "Keyword", "query": "Keyword", "Top queries": "Keyword", "queries": "Keyword", "Keywords": "Keyword", "keywords": "Keyword", "Search terms report": "Keyword"}, inplace=True)

        if "Keyword" not in df.columns:
            st.error("Error! Please make sure your CSV or XLSX file contains a column named 'Keyword'!")
        else:
            st.write(f"Total rows: {len(df)}")
            st.write(f"Number of unique keywords: {df['Keyword'].nunique()}")
            
            df['Keyword'] = df['Keyword'].astype(str)
            
            st.write("Sample of the data (first 5 rows):")
            st.write(df['Keyword'].head())

            sample_keywords = df['Keyword'].sample(min(100, len(df))).tolist()
            detected_languages = [detect_language(keyword) for keyword in sample_keywords]
            main_language = max(set(detected_languages), key=detected_languages.count)

            st.write(f"Detected main language: {main_language}")
            st.write("Other detected languages: " + ', '.join(set(detected_languages) - {main_language, 'unknown'}))

            # Geocode keywords
            geolocator = Nominatim(user_agent="keyword_clustering_app")
            df['Latitude'], df['Longitude'] = zip(*df['Keyword'].apply(lambda x: geocode_keyword(geolocator, x)))

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
                
                # Add geographical information to embeddings
                geo_data = df[df['Keyword'].isin(corpus_sentences)][['Latitude', 'Longitude']].values
                geo_data = np.nan_to_num(geo_data)  # Replace NaN with 0
                geo_weight = 0.5  # Adjust this weight to control the influence of geographical data
                combined_embeddings = np.hstack([corpus_embeddings.cpu().numpy(), geo_weight * geo_data])
                
                if clustering_method == "Community Detection":
                    clusters = util.community_detection(torch.tensor(combined_embeddings), min_community_size=min_cluster_size, threshold=cluster_accuracy)
                    cluster_labels = np.array([i for i, cluster in enumerate(clusters) for _ in cluster])
                elif clustering_method == "Agglomerative":
                    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
                    cluster_labels = clustering_model.fit_predict(combined_embeddings)
                elif clustering_method == "K-means":
                    clustering_model = KMeans(n_clusters=n_clusters)
                    cluster_labels = clustering_model.fit_predict(combined_embeddings)

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

                df = df[['Cluster Name', 'Keyword', 'Latitude', 'Longitude']]

                df.sort_values(["Cluster Name", "Keyword"], ascending=[True, True], inplace=True)

                uncluster_percent = (remaining / len(df)) * 100
                clustered_percent = 100 - uncluster_percent
                st.write(f"{clustered_percent:.2f}% of rows clustered successfully!")
                st.write(f"Number of iterations: {iterations}")
                st.write(f"Total unclustered keywords: {remaining}")

                st.write(f"Number of clusters: {len(df['Cluster Name'].unique())}")

                result_df = df.groupby('Cluster Name')['Keyword'].apply(', '.join).reset_index()
                result_df.columns = ['Cluster', 'Keywords']

                st.write(result_df)

                # 3D visualization
                embeddings = model.encode(df['Keyword'].tolist(), batch_size=256, show_progress_bar=True)
                geo_data = df[['Latitude', 'Longitude']].values
                geo_data = np.nan_to_num(geo_data)  # Replace NaN with 0
                combined_data = np.hstack([embeddings, geo_weight * geo_data])

                pca = PCA(n_components=3)
                embeddings_3d = pca.fit_transform(combined_data)

                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=embeddings_3d[:, 0],
                    y=embeddings_3d[:, 1],
                    z=embeddings_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df['Cluster Name'].astype('category').cat.codes,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=df['Keyword'],
                    hoverinfo='text'
                )])

                fig_3d.update_layout(
                    width=1200,
                    height=675,
                    title='Keyword Embeddings in 3D Space (with Geographical Context)',
                    scene=dict(
                        xaxis_title='Dimension 1',
                        yaxis_title='Dimension 2',
                        zaxis_title='Dimension 3'
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )

                st.plotly_chart(fig_3d, use_container_width=True)

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

                st.write("Note: This tool now incorporates geographical context for improved clustering of location-based keywords.")

    except pd.errors.EmptyDataError:
        st.error("EmptyDataError: No columns to parse from file. Please upload a valid CSV or XLSX file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please check your data and try again.")
