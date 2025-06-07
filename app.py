import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras.metrics import Precision, Recall
import numpy as np
from sklearn.cluster import KMeans
import base64


# Load dataset directly from file path
df = pd.read_csv(r"C:\Users\vodna\OneDrive\Desktop\inno\DL\Match_NoMatch\data.csv")

# Streamlit config
st.set_page_config(page_title="💞 Match Prediction App", layout="centered")
st.markdown("<h1 style='text-align: center; color: #e91e63;'>💞 Match Prediction App</h1>", unsafe_allow_html=True)

# Drop unwanted columns
if "Timestamp" in df.columns and "Email Address" in df.columns:
    df.drop(columns=["Timestamp", "Email Address"], inplace=True)

# Show raw data inside expander
with st.expander("📋 Click to view sample of raw data"):
    st.dataframe(df.head(), use_container_width=True)


# Label Encoding
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and labels (last column = label)
obj = KMeans(n_clusters=2,n_init=5).fit(df)
clusters = obj.labels_
df["labels"] = clusters
x = df.drop(columns="labels",axis=1)
y = df["labels"]


# Build ANN
model = keras.Sequential([
    keras.layers.Input(shape=(x.shape[1],)),
    keras.layers.Dense(8, kernel_initializer=keras.initializers.GlorotNormal(seed=42)),
    keras.layers.PReLU(),
    keras.layers.Dense(3, kernel_initializer=keras.initializers.GlorotNormal(seed=42),
                       kernel_regularizer=keras.regularizers.L2()),
    keras.layers.PReLU(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid", kernel_initializer=keras.initializers.HeNormal(seed=42))
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", Precision(), Recall()])


# Train model

if st.button("🚀 Train Model"):
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True, mode="min"
    )

    with st.spinner("⏳ Training in progress..."):
        history = model.fit(x, y, validation_split=0.2, batch_size=8, epochs=30,
                            callbacks=[checkpoint_cb], verbose=0)
    st.success("✅ Model training completed successfully!")

# Prediction section
st.markdown("---")
st.markdown("<h3 style='color: #FF4500;'>💌 Make a Prediction</h3>", unsafe_allow_html=True)

user_inputs = {}
for col in x.columns:
    options = encoders[col].classes_
    st.markdown(f"<h5 style='color: #4a148c; padding-bottom: 0.01cm;margin-top: 30px;'>💝 {col}:</h5>", unsafe_allow_html=True)
    selected = st.selectbox("", options)
    user_inputs[col] = encoders[col].transform([selected])[0]


def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_path):
    encoded = get_base64(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: 1800px 800px;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

if st.button("💜 Predict Match"):
    input_df = pd.DataFrame([user_inputs.values()], columns=user_inputs.keys())
    prediction = model.predict(input_df)[0][0]
    if prediction >= 0.5:
        st.success(f"💘 It's a Match! (Confidence: {round(prediction * 100, 2)}%)")
        set_background("icegif-1020.gif")
    else:
        st.warning(f"💔 No Match. (Confidence: {round((prediction) * 100, 2)}%)")
        set_background("200w.gif")


