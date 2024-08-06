import streamlit as st
import pandas as pd
import folium
from streamlit_option_menu import option_menu
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from streamlit_folium import folium_static
from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO
import json

# Hide Streamlit style
hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

st.set_page_config(
    page_title="SMART",
    page_icon="memo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/benedictus-briatore-ananta-ba921b281/',
        'Report a bug': "https://github.com/benedictusbriatoreananta/dashboard",
        'About': "## A 'Badspot Prediction Tool' by Benedictus Briatore Ananta"
    }
)

st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """, unsafe_allow_html=True
)

selected = "Menu Utama"

with st.sidebar:
    selected = option_menu(
        menu_title="Badspot Prediction",
        options=["Menu Utama", "Predictions", "Contributors"],
        icons=["house", "upload", "people"],
        menu_icon="broadcast tower",
        default_index=0,
    )

# Upload the credential file
st.sidebar.markdown("### Upload credential.json file")
uploaded_credential = st.sidebar.file_uploader("Choose the credential.json file", type="json")

if uploaded_credential:
    try:
        credentials = service_account.Credentials.from_service_account_info(
            json.load(uploaded_credential)
        )
        client = storage.Client(credentials=credentials)
        bucket_name = 'model_skripsi_ml'  # Update with your GCS bucket name
        bucket = client.bucket(bucket_name)

        st.sidebar.success("Credential file uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading credential file: {e}")

# Define functions for GCS
def upload_to_gcs(file, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file)
    st.success(f'File uploaded to GCS bucket {bucket_name} as {destination_blob_name}.')

def download_from_gcs(source_blob_name):
    blob = bucket.blob(source_blob_name)
    file_content = blob.download_as_bytes()
    return BytesIO(file_content)

# Define load_model function
def load_model_and_scaler():
    model_path = "svc_model.pkl"  # Update with the correct GCS path
    scaler_path = "scaler.pkl"    # Update with the correct GCS path
    encoder_path = "label_encoder.pkl"  # Update with the correct GCS path

    try:
        model_blob = download_from_gcs(model_path)
        scaler_blob = download_from_gcs(scaler_path)
        encoder_blob = download_from_gcs(encoder_path)

        model = joblib.load(model_blob)
        scaler = joblib.load(scaler_blob)
        label_encoder = joblib.load(encoder_blob)

        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading the model, scaler, or encoder: {e}")
        return None, None, None

# Define preprocess_data function
def preprocess_data(data, feature_names, scaler, label_encoder):
    if not all(column in data.columns for column in feature_names):
        st.error("The input data does not contain all required columns.")
        return None

    try:
        data['Cat'] = label_encoder.transform(data['Cat'])
    except Exception as e:
        st.error(f"Error encoding 'Cat' column: {e}")
        return None

    numeric_cols = feature_names
    data = data[numeric_cols]

    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)

    data = scaler.transform(data)

    return data

# Define make_predictions function
def make_predictions(model, data, scaler, label_encoder):
    try:
        feature_names = ['Longitude', 'Latitude', 'PCI LTE', 'TAC', 'MCC', 'MNC', 'RSRP', 'RSRQ', 'DL EARFCN', 'Cat']

        data_preprocessed = preprocess_data(data, feature_names, scaler, label_encoder)
        if data_preprocessed is None:
            return None

        predictions = model.predict(data_preprocessed)
        data['Prediction'] = predictions

        # Logika untuk menentukan "Badspot" dan "Non-Badspot"
        data['Prediction'] = data.apply(
            lambda x: 0 if x['RSRP'] >= -80 and x['RSRQ'] >= -10 else (1 if x['Prediction'] == 1 else 0),
            axis=1
        )

        st.write("Predictions:")
        st.write(data[['Longitude', 'Latitude', 'RSRP', 'RSRQ', 'Prediction']])

        return data
    except Exception as e:
        st.error(f"Error making predictions: {e}")
       
