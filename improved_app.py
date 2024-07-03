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

# Function to calculate cluster coherence
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

# Configuration
st.title("Semantic Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

# Sidebar for clustering method selection
clustering_method = st.sidebar.selectbox(
    "Select Clustering Method",
    ["Community Detection", "Agglomerative", "K-means"]
)

# Conditional inputs based on clustering method
if clustering_method == "Community Detection":
    cluster_accuracy = st.slider("Cluster Accuracy (0-100)", 0, 100, 80) / 100
    min_cluster_size = st.number_input("Minimum Cluster Size", min_value=1, max_value=100, value=3)
elif clustering_method == "Agglomerative":
    distance_threshold = st.sidebar.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
elif clustering_method == "K-means":
    n_clusters = st.number_input("Number of Clusters for K-means", min_value=2, max_value=100, value=5)

transformer = st.selectbox("Select Transformer Model", ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-mpnet-base-v2'])

uploaded_file = st.file_uploader("Upload Keyword CSV or XLSX", type=["csv", "xlsx"])

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
            # Data validation
            st.write(f"Total rows: {len(df)}")
            st.write(f"Number of unique keywords: {df['Keyword'].nunique()}")
            st.write("Data types in 'Keyword' column:")
            st.write(df['Keyword'].apply(type).value_counts())
            
            # Convert all values in the 'Keyword' column to strings
            df['Keyword'] = df['Keyword'].astype(str)
            
            # Display a sample of the data
            st.write("Sample of the data (first 5 rows):")
            st.write(df.head())

            # Clustering Process
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
                
                # Ensure corpus_embeddings is 2D
                if len(corpus_embeddings.shape) == 1:
                    corpus_embeddings = corpus_embeddings.reshape(1, -1)
                
                if clustering_method == "Community Detection":
                    clusters = util.community_detection(corpus_embeddings, min_community_size=min_cluster_size, threshold=cluster_accuracy)
                    cluster_labels = np.array([i for i, cluster in enumerate(clusters) for _ in cluster])
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

                df_new = pd.DataFrame(None)
                df_new['Cluster Name'] = cluster_name_list
                df_new["Keyword"] = corpus_sentences_list

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

                df['Length'] = df['Keyword'].astype(str).map(len)
                df = df.sort_values(by="Length", ascending=True)

                df['Cluster Name'] = df.groupby('Cluster Name')['Keyword'].transform('first')
                df.sort_values(['Cluster Name', "Keyword"], ascending=[True, True], inplace=True)

                df['Cluster Name'] = df['Cluster Name'].fillna("no_cluster")

                del df['Length']

                col = df.pop("Keyword")
                df.insert(0, col.name, col)

                col = df.pop('Cluster Name')
                df.insert(0, col.name, col)

                df.sort_values(["Cluster Name", "Keyword"], ascending=[True, True], inplace=True)

                uncluster_percent = (remaining / len(df)) * 100
                clustered_percent = 100 - uncluster_percent
                st.write(f"{clustered_percent:.2f}% of rows clustered successfully!")
                st.write(f"Number of iterations: {iterations}")
                st.write(f"Total unclustered keywords: {remaining}")

                # Print number of clusters
                st.write(f"Number of clusters: {len(df['Cluster Name'].unique())}")

                # Calculate cluster coherences
                if len(corpus_embeddings) > 1:
                    cluster_coherences = calculate_cluster_coherence(corpus_embeddings.cpu().numpy(), cluster_labels)
                    overall_coherence = np.mean(cluster_coherences)

                    st.write(f"Overall Clustering Coherence: {overall_coherence:.4f}")
                    st.write("Cluster Coherences:")
                    for i, coherence in enumerate(cluster_coherences):
                        st.write(f"Cluster {i+1}: {coherence:.4f}")
                else:
                    st.write("Coherence: Not applicable (insufficient data)")

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

                # Display unclustered keywords and add download button
                if remaining > 0:
                    st.write("Unclustered Keywords:")
                    st.write(list(corpus_set))
                    
                    # Create a DataFrame of unclustered keywords
                    unclustered_df = pd.DataFrame(list(corpus_set), columns=['Unclustered Keyword'])
                    
                    # Download button for unclustered keywords
                    csv_data_unclustered = unclustered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Unclustered Keywords",
                        data=csv_data_unclustered,
                        file_name="Unclustered_Keywords.csv",
                        mime="text/csv"
                    )

    except pd.errors.EmptyDataError:
        st.error("EmptyDataError: No columns to parse from file. Please upload a valid CSV or XLSX file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please check your data and try again.")
