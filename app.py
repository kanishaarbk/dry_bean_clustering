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
model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", value=28395.0)
    perimeter = st.number_input("Perimeter", value=610.291)
    major = st.number_input("MajorAxisLength", value=208.1781167)
    minor = st.number_input("MinorAxisLength", value=173.888747)
    aspect = st.number_input("AspectRation", value=1.197191424)
    eccentricity = st.number_input("Eccentricity", value=0.5498121871)
    convex = st.number_input("ConvexArea", value=28715.0)
    equiv = st.number_input("EquivDiameter", value=190.1410973)

with col2:
    extent = st.number_input("Extent", value=0.7639225182)
    solidity = st.number_input("Solidity", value=0.9888559986)
    roundness = st.number_input("Roundness", value=0.9580271263)
    compactness = st.number_input("Compactness", value=0.9133577548)
    sf1 = st.number_input("ShapeFactor1", value=0.007331506135, format="%.10f")
    sf2 = st.number_input("ShapeFactor2", value=0.003147289167, format="%.10f")
    sf3 = st.number_input("ShapeFactor3", value=0.8342223882)
    sf4 = st.number_input("ShapeFactor4", value=0.998723889)
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
