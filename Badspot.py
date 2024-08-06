import streamlit as st
import pandas as pd
import folium
from streamlit_option_menu import option_menu
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from streamlit_folium import folium_static
from google.cloud import storage
import json
import os

# Set environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/credentials.json'

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

# Define load_model function from GCS
def load_model(bucket_name, model_path, scaler_path, encoder_path, credentials_json):
    storage_client = storage.Client.from_service_account_info(credentials_json)
    bucket = storage_client.bucket(bucket_name)

    def download_blob_to_file(blob_name, file_path):
        blob = bucket.blob(blob_name)
        blob.download_to_filename(file_path)

    local_model_path = "local_model.pkl"
    local_scaler_path = "local_scaler.pkl"
    local_encoder_path = "local_encoder.pkl"

    download_blob_to_file(model_path, local_model_path)
    download_blob_to_file(scaler_path, local_scaler_path)
    download_blob_to_file(encoder_path, local_encoder_path)

    model = joblib.load(local_model_path)
    scaler = joblib.load(local_scaler_path)
    label_encoder = joblib.load(local_encoder_path)

    return model, scaler, label_encoder

# Define preprocess_data function
def preprocess_data(data, feature_names, scaler, label_encoder):
    if not all(column in data.columns for column in feature_names):
        st.error("The input data does not contain all required columns.")
        return None

    data['Cat'] = label_encoder.transform(data['Cat'])

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

        data['Prediction'] = data.apply(
            lambda x: 0 if x['RSRP'] >= -80 and x['RSRQ'] >= -10 else (1 if x['Prediction'] == 1 else 0),
            axis=1
        )

        return data
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None

# Membuat Peta
def display_predictions_on_map(predictions):
    if predictions is None or predictions.empty:
        st.error("No predictions to display on the map.")
        return
    
    map_center = [predictions['Latitude'].mean(), predictions['Longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=10)

    for i, row in predictions.iterrows():
        location = [row['Latitude'], row['Longitude']]
        popup_text = f"RSRP: {row['RSRP']}<br>RSRQ: {row['RSRQ']}<br>Prediction: {'Badspot' if row['Prediction'] == 1 else 'Non-Badspot'}"
        color = 'red' if row['Prediction'] == 1 else 'green'
        folium.Marker(location, popup=popup_text, icon=folium.Icon(color=color)).add_to(m)

    folium_static(m)

# Home tab
if selected == "Menu Utama":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            <h1>Badspot Prediction <i class="fas fa-broadcast-tower"></i></h1>
            """, unsafe_allow_html=True
        )
        st.divider()
        st.header("About :memo:")
        st.markdown('''
        ####
        Selamat datang di situs Prediksi Badspot. Platform kami dirancang untuk lembaga pemerintah, Kementerian Komunikasi dan Informatika, dalam mengelola dan memprediksi potensi masalah yang mungkin timbul dalam domainnya masing-masing. Dengan memanfaatkan analisis prediktif tingkat lanjut, kami dapat membantu mengidentifikasi area yang berisiko mengalami inefisiensi operasional, ancaman keamanan, atau gangguan komunikasi, sehingga memungkinkan intervensi dan dukungan tepat waktu.

        Misi kami adalah mendukung institusi pendidikan dalam menciptakan lingkungan belajar yang aman dan produktif. Dengan memberikan tanda peringatan dini dan wawasan yang dapat ditindaklanjuti, kami bertujuan untuk memberdayakan pendidik dan administrator untuk mengambil langkah proaktif dalam mendukung lembaga pemerintah.
        
        Terima kasih telah menggunakan platform kami. Kami berkomitmen untuk melakukan perbaikan terus-menerus dan menyambut masukan apa pun yang Anda miliki.
        ''')
        
        st.markdown("#### `Get Started Now!`")

# Prediction tab
if selected == "Predictions":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Menu Prediction ⚡")
        st.subheader("Beri data input di bawah ini 👇🏻")
        st.divider()
        st.markdown("##### _Disini kita menggunakan algoritma <span style='color:yellow'>SVM (Support Vector Machine)🤖</span> Algoritma Pembelajaran Mesin untuk membuat Model kami untuk memprediksi letak/titik yang akan terjadinya badspot_.", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.title("Upload Data for Prediction 🗃️")
        st.markdown("#### Upload the Excel file containing data for prediction..")
        uploaded_file = st.file_uploader("Choose an Excel file 📂", type=['xlsx'])
        predict_button = st.button("Predict the Placement ⚡")

        if uploaded_file and predict_button:
            with st.spinner('Processing...'):
                try:
                    input_data = pd.read_excel(uploaded_file)
                    st.markdown("* ## Input Dataframe ⬇️")
                    st.write(input_data)
                    required_columns = ['Longitude', 'Latitude', 'PCI LTE', 'TAC', 'MCC', 'MNC', 'RSRP', 'RSRQ', 'DL EARFCN', 'Cat']
                    if not all(column in input_data.columns for column in required_columns):
                        st.error("The uploaded file does not contain all required columns.")
                    else:
                        input_data = input_data[required_columns]

                        bucket_name = 'badspot-predict'
                        model_path = 'models/svc_model.pkl'
                        scaler_path = 'models/scaler.pkl'
                        encoder_path = 'models/label_encoder.pkl'

                        credentials_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                        if not credentials_json_str:
                            st.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set or is empty.")
                        else:
                            credentials_json = json.loads(credentials_json_str)

                            model, scaler, label_encoder = load_model(bucket_name, model_path, scaler_path, encoder_path, credentials_json)

                            predictions = make_predictions(model, input_data, scaler, label_encoder)
                            if predictions is not None:
                                st.success("Predictions made successfully.")
                                st.write(predictions)
                                display_predictions_on_map(predictions)
                            else:
                                st.error("Failed to make predictions.")
                except Exception as e:
                    st.error(f"Error processing the file: {e}")

# Contributors tab
if selected == "Contributors":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Meet Our Contributors 👨🏻‍💻")
        st.divider()
        st.markdown('''
        ####
        - **Benedictus Briatore Ananta**
    
        ### Acknowledgments:
        We would like to express our gratitude to all those who have contributed to this project. Special thanks to our mentors, peers, and the open-source community for their invaluable support and contributions.
        ''')
