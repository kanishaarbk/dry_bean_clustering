import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Dry Bean Cluster Predictor",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Cluster Predictor")
st.write("Enter the bean measurements below and click Predict.")

# ---------------- LOAD FILES ----------------
model = pickle.load(open("k_meansmodel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", min_value=0.0)
    perimeter = st.number_input("Perimeter", min_value=0.0)
    major = st.number_input("MajorAxisLength", min_value=0.0)
    minor = st.number_input("MinorAxisLength", min_value=0.0)
    aspect = st.number_input("AspectRation", min_value=0.0)
    eccentricity = st.number_input("Eccentricity", min_value=0.0)
    convex = st.number_input("ConvexArea", min_value=0.0)
    equiv = st.number_input("EquivDiameter", min_value=0.0)

with col2:
    extent = st.number_input("Extent", min_value=0.0)
    solidity = st.number_input("Solidity", min_value=0.0)
    roundness = st.number_input("Roundness", min_value=0.0)
    compactness = st.number_input("Compactness", min_value=0.0)
    sf1 = st.number_input("ShapeFactor1", min_value=0.0)
    sf2 = st.number_input("ShapeFactor2", min_value=0.0)
    sf3 = st.number_input("ShapeFactor3", min_value=0.0)
    sf4 = st.number_input("ShapeFactor4", min_value=0.0)

# ---------------- PREDICTION ----------------
if st.button("Predict Cluster", use_container_width=True):

    input_df = pd.DataFrame([[
        area,
        perimeter,
        major,
        minor,
        aspect,
        eccentricity,
        convex,
        equiv,
        extent,
        solidity,
        roundness,
        compactness,
        sf1,
        sf2,
        sf3,
        sf4
    ]], columns=columns)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Cluster : {prediction}")

    if prediction == 0:
        st.info("🌱 This bean belongs to Cluster 0")
    elif prediction == 1:
        st.info("🌱 This bean belongs to Cluster 1")
    else:
        st.info(f"🌱 This bean belongs to Cluster {prediction}")
