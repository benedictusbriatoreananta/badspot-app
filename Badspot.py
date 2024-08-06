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

# Fungsi untuk menghubungkan ke GCP
def connect_to_gcp():
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if credentials_json is None:
        st.error("Environment variable 'GOOGLE_APPLICATION_CREDENTIALS_JSON' is not set.")
        return None
    
    credentials_dict = json.loads(credentials_json)
    client = storage.Client.from_service_account_info(credentials_dict)
    return client

# Contoh penggunaan fungsi connect_to_gcp
client = connect_to_gcp()
if client is None:
    st.stop()  # Stop the app if the client is not created

bucket_name = 'badspot_predict'
bucket = client.get_bucket(bucket_name)

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

                        credentials_json = {
                          "type": "service_account",
                          "project_id": "badspot-predict",
                          "private_key_id": "f845ec177bda10934cc4c706609ccf8cc72abc62",
                          "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDCmZl/DmpRHdaV\nMDPFLLmS76DLtOWgNAYAcLNgVItRx/3Zh9nY/usARt8/yNUXOmpjhVyJqpcTB6j5\n8mJsOof2vN0c2raqgU3fv3QgMiwhgzldZOjht52jgNgg3sJQIQCqrIuGUhaK6qFS\n8+HCLvPgA4pfHtc1mpOXCB8ntUheySCLqLMprWQSrOqXzXpLznZY6Thy3wEZ1nZS\n7Mn0xEcV4xu113D+JPi/GTCD5e622nuCk/vegieKWOI1/3ZAHksphDowuRKAXp5B\nbhZ60Te5P6SsGKUzAc5u5+Ds1cisEtUWUh+Dbx6EmuVqyayW/buXnGonrixFD76m\n4JbG/H5JAgMBAAECggEAO1qIA8XMzaLxMR27nJcwns1x2M/6/asAbZ/yRh5jQ/Vg\nyrlxy4qs+K+lJhpGTEhn7KTffanHXGmIs3unY2VS1QHz2vomnsjDjMwjSjKjXTFe\nDMtbcIUFkroYsaXf1HrMbkIkaDqfX42C2P9Dy2twvNG3oZ2RfGsCOx2iCwVy5l2e\n5nE3H+UtpENUjlQmGUdvx0Gt4btaTAXoB6u8uGVodUka2w6uJQRxNj7rUjT+AYOF\n3rsDcSoAkgTBiFQD25gixmrlpoGI26i5R7ZgkdZWzhKB55eTinV08TAMyPB6hqJu\nNZAzsgY17VU8l2iqILzAroWxkBOzoSUw1x9eXUdKFQKBgQD72nVE4sIiZEYaanuj\n8H4vMf7C0dvidyU1lMzkfOPBOoBoHIcC7HB9ibz5lolY9/+1Qpb0r7CI1Ty+moXw\n+P9MB90cM2LvSfFeiAn5OD1cHRE8uLprVumz/uDdyDQj+jpyFPWARtsNJGnD2IlE\n/qz919iJ6UvLTd+bW7B3lbaKiwKBgQDFzdK69tUoVJlW3DqwhOq0XJJOjKVfKleN\nO1RI86E+wZW/pfKneWVoXEd1uzfTdMjgyGxeyylaAp0tAbGJR1N5ovT4ZVCCtlYH\n0Nujigd8A1aH4+DRV0xGJxDV6+sfy2ifJbPzLxS6YiLxIHUs9Y1dSGye109QFn2i\nK0hvqeT4+wKBgE86vVOLZnk9RFSBFR5QxNGCD9wn+t12j+0YP5DNvTSHe+fEubBw\nwz6q+xklg1XKxtW1+hlFv+p78p0frW1OV7oKa0O44rHWeCk98K1HRP5aYpbQokys\nTd8DGqiKl+SNjp9e+pB9OeIbh25GC4D6AV/l7EgObXqqp+KO6KfmIEPbAoGAeNiJ\nYS5CBhTBZd7AgG2EEtLnk9O7iMuOl7tif/tQTM3qVh7lg8nX2Y1fHx9VOPwFacco\n5jUKu3ITYpbBR1RrPEoBjewf5uvM3ZONTHmcnvhPGlBvXpYOBGIDmB49FLyp22km\nanEeIcyo+lXRILmYNyBzNAvo6c4DzVrSM15BhJcCgYEAsNbqypWEAYzN3d0YKy0d\n3yqYPzGct2y0icT5VLmfWdWdMfSluVtrOf9B99mySKe3eXD29Hqtdn3uOWa/ZPEn\nB5fFNO7GNh14kKBcPOq3gJkjAKaCSvZJ6V/AIigiQLoAhnBJX2L8uF21gaGV7AmC\nAWClSA+65FAVRUJOtDRwW3s=\n-----END PRIVATE KEY-----\n",
                          "client_email": "streamlit-account-service@badspot-predict.iam.gserviceaccount.com",
                          "client_id": "114756292400923299842",
                          "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                          "token_uri": "https://oauth2.googleapis.com/token",
                          "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                          "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/streamlit-account-service%40badspot-predict.iam.gserviceaccount.com",
                          "universe_domain": "googleapis.com"
                        }

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
        - **Benedictus Briatore Ananta**: Lead Data Scientist
        - **Team Member 2**: Role
        - **Team Member 3**: Role
        - **Team Member 4**: Role
        
        ### Acknowledgments:
        We would like to express our gratitude to all those who have contributed to this project. Special thanks to our mentors, peers, and the open-source community for their invaluable support and resources.
        ''')
