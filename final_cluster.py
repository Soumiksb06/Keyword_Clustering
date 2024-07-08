import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
from langdetect import detect_langs
from iso639 import languages

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

st.title("Geographical-Aware Keyword Clustering Tool")
st.write("Upload a CSV or XLSX file containing keywords for clustering.")

clustering_method = st.selectbox(
    "Select Clustering Method",
    ["Agglomerative", "K-means"]
)

if clustering_method == "Agglomerative":
    distance_threshold = st.number_input("Distance Threshold for Agglomerative Clustering", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
else:
    n_clusters = st.number_input("Number of Clusters for K-means", min_value=2, max_value=1000, value=50)

transformer_model = 'distiluse-base-multilingual-cased-v2'

uploaded_file = st.file_uploader("Upload Keyword CSV or XLSX", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.write("File loaded successfully!")

        # Add dropdown to select keyword column
        keyword_column = st.selectbox("Select the column containing keywords", df.columns)

        if keyword_column not in df.columns:
            st.error(f"Error! The selected column '{keyword_column}' is not in the dataframe!")
        else:
            df['Keyword'] = df[keyword_column].astype(str)
            
            st.write(f"Total rows: {len(df)}")
            st.write(f"Number of unique keywords: {df['Keyword'].nunique()}")
            
            st.write("Sample of the data (first 5 rows):")
            st.write(df['Keyword'].head())

            sample_keywords = df['Keyword'].sample(min(100, len(df))).tolist()
            detected_languages = [detect_language(keyword) for keyword in sample_keywords]
            main_language = max(set(detected_languages), key=detected_languages.count)

            st.write(f"Detected main language: {main_language}")
            st.write("Other detected languages: " + ', '.join(set(detected_languages) - {main_language, 'Unknown Language'}))

            with st.spinner("Geocoding keywords..."):
                geolocator = Nominatim(user_agent="keyword_clustering_app")
                df['Latitude'], df['Longitude'] = zip(*df['Keyword'].apply(lambda x: geocode_keyword(geolocator, x)))

            model = SentenceTransformer(transformer_model)
            
            with st.spinner("Encoding keywords..."):
                embeddings = model.encode(df['Keyword'].tolist(), show_progress_bar=False)

            geo_data = df[['Latitude', 'Longitude']].values
            geo_data = np.nan_to_num(geo_data)  # Replace NaN with 0
            geo_weight = 0.5  # Adjust this weight to control the influence of geographical data
            combined_data = np.hstack([embeddings, geo_weight * geo_data])

            with st.spinner("Clustering keywords..."):
                if clustering_method == "Agglomerative":
                    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
                else:
                    clustering_model = KMeans(n_clusters=n_clusters)
                
                cluster_labels = clustering_model.fit_predict(combined_data)

            df['Cluster'] = cluster_labels

            st.write(f"Number of clusters: {len(df['Cluster'].unique())}")

            result_df = df.groupby('Cluster')['Keyword'].apply(list).reset_index()
            result_df['Cluster_Size'] = result_df['Keyword'].apply(len)
            result_df = result_df.sort_values('Cluster_Size', ascending=False)

            st.write("Cluster Summary:")
            for _, row in result_df.iterrows():
                st.write(f"Cluster {row['Cluster']} (Size: {row['Cluster_Size']}):")
                st.write(", ".join(row['Keyword'][:10]) + ("..." if len(row['Keyword']) > 10 else ""))
                st.write("---")

            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Clustered Keywords",
                data=csv_data,
                file_name="Clustered_Keywords.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please check your data and try again.")
