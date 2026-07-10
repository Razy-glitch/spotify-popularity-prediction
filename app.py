import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load model
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# Load dataset
df = pd.read_csv("spotify-tracks-dataset-detailed.csv")


st.set_page_config(
    page_title="Spotify Popularity Prediction",
    layout="wide"
)

st.markdown(
    """
    <style>

    .main {
        background-color: #fafafa;
    }

    h1 {
        color: #1DB954;
    }

    h2 {
        color: #191414;
    }

    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 45px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎵 Spotify Popularity Prediction")

st.caption(
    "Machine Learning Classification using Random Forest Algorithm"
)
st.write(
    "Prediksi popularitas lagu Spotify berdasarkan karakteristik audio menggunakan Machine Learning."
)

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg",
    width=100
)

st.sidebar.title(
    "Spotify ML App"
)

st.sidebar.write(
    "Predict song popularity based on audio features"
)
menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard EDA",
        "Prediksi Lagu",
        "Evaluasi Model",
        "Interpretasi Model"
    ]
)


if menu == "Dashboard EDA":

    st.header("📊 Dashboard Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Lagu",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Jumlah Fitur",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Model",
            "Random Forest"
        )


    st.subheader("Distribusi Popularity")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        df["popularity"],
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title("Distribusi Popularity Lagu Spotify")
    ax.set_xlabel("Popularity Score")
    ax.set_ylabel("Jumlah Lagu")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

elif menu == "Prediksi Lagu":

    st.header("🎵 Prediksi Popularitas Lagu")

    st.write(
        "Masukkan karakteristik audio lagu untuk memprediksi apakah lagu termasuk populer atau tidak."
    )


    col1, col2 = st.columns(2)


    with col1:

        danceability = st.slider(
            "Danceability",
            0.0,
            1.0,
            0.5
        )

        energy = st.slider(
            "Energy",
            0.0,
            1.0,
            0.5
        )

        loudness = st.number_input(
            "Loudness",
            value=-7.0
        )

        speechiness = st.slider(
            "Speechiness",
            0.0,
            1.0,
            0.05
        )

        acousticness = st.slider(
            "Acousticness",
            0.0,
            1.0,
            0.5
        )

        instrumentalness = st.slider(
            "Instrumentalness",
            0.0,
            1.0,
            0.0
        )


    with col2:

        liveness = st.slider(
            "Liveness",
            0.0,
            1.0,
            0.2
        )

        valence = st.slider(
            "Valence",
            0.0,
            1.0,
            0.5
        )

        tempo = st.number_input(
            "Tempo",
            value=120.0
        )

        duration_ms = st.number_input(
            "Duration (ms)",
            value=200000
        )

        explicit = st.selectbox(
            "Explicit",
            [0,1]
        )


    if st.button("Prediksi"):

        input_data = pd.DataFrame(
            [[
                danceability,
                energy,
                loudness,
                speechiness,
                acousticness,
                instrumentalness,
                liveness,
                valence,
                tempo,
                duration_ms,
                explicit
            ]],
            columns=features
        )


        input_scaled = scaler.transform(
            input_data
        )


        prediction = model.predict(
            input_scaled
        )


        if prediction[0] == 1:
            st.success(
                "🎵 Lagu diprediksi POPULER"
            )
        else:
            st.warning(
                "🎧 Lagu diprediksi TIDAK POPULER"
            )
        
elif menu == "Evaluasi Model":

    st.header("📈 Evaluasi Model Random Forest")

    st.write(
        "Evaluasi dilakukan menggunakan hasil pengujian model Random Forest pada data testing."
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric(
            "Accuracy",
            "78.01%"
        )

    with col2:
        st.metric(
            "Precision",
            "76.45%"
        )

    with col3:
        st.metric(
            "Recall",
            "70.21%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "73.19%"
        )

elif menu == "Interpretasi Model":

    st.header("🧠 Interpretasi Model Terbaik")

    st.write(
        "Model terbaik yang dipilih adalah Random Forest karena memberikan performa tertinggi dibandingkan model lainnya."
    )


    st.subheader("Performa Model Terbaik")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "F1 Score",
            "73.19%"
        )


    st.subheader("Feature Importance")


    importance = pd.DataFrame({
        "Feature": [
            "duration_ms",
            "acousticness",
            "valence",
            "danceability",
            "energy",
            "tempo",
            "loudness",
            "speechiness",
            "liveness",
            "instrumentalness",
            "explicit"
        ],
        "Importance": [
            0.105878,
            0.105699,
            0.103702,
            0.103009,
            0.101711,
            0.101651,
            0.101305,
            0.101003,
            0.095214,
            0.074105,
            0.006723
        ]
    })


    fig, ax = plt.subplots(figsize=(7,4))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title(
        "Feature Importance - Random Forest"
    )

    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)


    st.info(
        """
        Berdasarkan hasil feature importance, fitur yang paling berpengaruh
        terhadap prediksi popularitas lagu adalah duration_ms, acousticness,
        valence, danceability, dan energy.

        Hal ini menunjukkan bahwa karakteristik audio memiliki pengaruh lebih
        besar dibandingkan atribut explicit dalam menentukan popularitas lagu.
        """
    )


    st.subheader("Confusion Matrix")


    cm = pd.DataFrame(
        [
            [10900,2100],
            [2893,6817]
        ],
        columns=[
            "Predicted 0",
            "Predicted 1"
        ],
        index=[
            "Actual 0",
            "Actual 1"
        ]
    )


    fig, ax = plt.subplots(figsize=(4,3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix Random Forest"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    