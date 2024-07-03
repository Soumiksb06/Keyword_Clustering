import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import chardet
from detect_delimiter import detect
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import colorsys
from sklearn.cluster import AgglomerativeClustering, KMeans

# Function to generate distinct colors in RGB format
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

# Function to perform Agglomerative Clustering
def perform_agglomerative_clustering(df, num_clusters):
    embeddings = np.random.randn(len(df), 3)  # Replace with actual embeddings
    clustering = AgglomerativeClustering(n_clusters=num_clusters)
    df['Cluster Name'] = clustering.fit_predict(embeddings)
    return df

# Function to perform K-Means Clustering
def perform_kmeans_clustering(df, num_clusters):
    embeddings = np.random.randn(len(df), 3)  # Replace with actual embeddings
    clustering = KMeans(n_clusters=num_clusters)
    df['Cluster Name'] = clustering.fit_predict(embeddings)
    return df

# Function to perform Community Detection
def perform_community_detection(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Keyword'].tolist(), batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    clusters = util.community_detection(embeddings)
    
    cluster_name_list = []
    for keyword, cluster in enumerate(clusters):
        for sentence_id in cluster:
            cluster_name_list.append(f"Cluster {keyword + 1}, #{len(cluster)} Elements")
    
    df['Cluster Name'] = cluster_name_list
    return df

# Configuration
st.title("Semantic Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

cluster_accuracy = st.slider("Cluster Accuracy (0-100)", 0, 100, 80) / 100
min_cluster_size = st.number_input("Minimum Cluster Size", min_value=1, max_value=100, value=3)
transformer = st.selectbox("Select Transformer Model", ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'roberta-base', 'bert-base-uncased', 'distilbert-base-uncased'])
uploaded_file = st.file_uploader("Upload Keyword CSV or XLSX", type=["csv", "xlsx"])

# Sidebar for choosing clustering method
clustering_method = st.sidebar.selectbox("Choose Clustering Method", ['Agglomerative', 'K-Means', 'HDBSCAN', 'Community Detection'])

if st.button("Start Clustering"):
    if uploaded_file:
        try:
            acceptable_confidence = 0.8

            # Read the uploaded file
            raw_data = uploaded_file.getvalue()
            detected = chardet.detect(raw_data)
            encoding_type = detected['encoding']
            if detected['confidence'] < acceptable_confidence:
                encoding_type = 'utf-8'
            try:
                text_data = raw_data.decode(encoding_type)
            except UnicodeDecodeError:
                encoding_type = 'ISO-8859-1'
                text_data = raw_data.decode(encoding_type)

            firstline = text_data.splitlines()[0]
            
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                delimiter_type = detect(firstline)
                if delimiter_type:
                    df = pd.read_csv(uploaded_file, encoding=encoding_type, delimiter=delimiter_type)
                else:
                    df = pd.read_csv(uploaded_file, encoding=encoding_type)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')

            st.write("File loaded successfully!")
            st.write(f"Detected encoding: '{encoding_type}'")

            df.rename(columns={"Search term": "Keyword", "keyword": "Keyword", "query": "Keyword", "Top queries": "Keyword", "queries": "Keyword", "Keywords": "Keyword", "keywords": "Keyword", "Search terms report": "Keyword"}, inplace=True)

            if "Keyword" not in df.columns:
                st.error("Error! Please make sure your CSV or XLSX file contains a column named 'Keyword'!")
            else:
                # Clustering Process based on selected method
                if clustering_method == 'Agglomerative':
                    num_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=100, value=5)
                    df = perform_agglomerative_clustering(df, num_clusters)
                elif clustering_method == 'K-Means':
                    num_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=100, value=5)
                    df = perform_kmeans_clustering(df, num_clusters)
                elif clustering_method == 'HDBSCAN':
                    df = perform_hdbscan_clustering(df)
                elif clustering_method == 'Community Detection':
                    df = perform_community_detection(df)

                df['Length'] = df['Keyword'].astype(str).map(len)
                df = df.sort_values(by="Length", ascending=True)

                df['Cluster Name'] = df['Cluster Name'].fillna("no_cluster")

                del df['Length']

                col = df.pop("Keyword")
                df.insert(0, col.name, col)

                col = df.pop('Cluster Name')
                df.insert(0, col.name, col)

                df.sort_values(["Cluster Name", "Keyword"], ascending=[True, True], inplace=True)

                st.write(f"Number of clusters: {len(df['Cluster Name'].unique())}")

                # Save results with only 'Cluster' and 'Keywords' columns
                result_df = df.groupby('Cluster Name')['Keyword'].apply(', '.join).reset_index()
                result_df.columns = ['Cluster', 'Keywords']

                # Display result dataframe
                st.write(result_df)

                # Generate 3D plot of clusters
                # Assuming you have numerical embeddings or features for each keyword
                # For demonstration, let's create random data
                np.random.seed(42)
                num_keywords = len(df['Keyword'])
                embeddings_3d = np.random.randn(num_keywords, 3)  # Replace with your actual 3D embeddings

                # Generate colors for clusters
                unique_clusters = df['Cluster Name'].unique()
                num_clusters = len(unique_clusters)
                cluster_colors = generate_colors(max(num_clusters, 100))

                # Create a figure for the 3D scatter plot
                fig_3d = go.Figure()

                for i, cluster in enumerate(unique_clusters):
                    cluster_data = df[df['Cluster Name'] == cluster]
                    fig_3d.add_trace(go.Scatter3d(
                        x=embeddings_3d[cluster_data.index, 0],
                        y=embeddings_3d[cluster_data.index, 1],
                        z=embeddings_3d[cluster_data.index, 2],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=f'rgb{cluster_colors[i]}',
                            opacity=0.8,
                        ),
                        text=cluster_data['Keyword'],  # Display keyword names on hover
                        hoverinfo='text',  # Show only text (keyword names) on hover
                        name=cluster
                    ))

                # Update layout for 3D scatter plot
                fig_3d.update_layout(
                    width=800,
                    height=700,
                    title='Keyword Clusters in 3D Space',
                    scene=dict(
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        zaxis_title='Z Axis'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0)
                )

                # Display 3D scatter plot using Streamlit
                st.plotly_chart(fig_3d, use_container_width=True)

                # Download button for clustered keywords
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

