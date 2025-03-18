import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image

# Ladda modellen med joblib
model = joblib.load("models/best_model_RF.pkl")

# Titel på appen
st.title("MNIST Klassificering med Random Forest")

st.write("Ladda upp en bild på en handskriven siffra och få en förutsägelse!")

# Funktion för bildförbehandling
def preprocess_image(image):
    # Konvertera till gråskala
    gray = np.array(image.convert("L"))
    
    # Omvandla till svartvitt
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Förstärka strecken
    kernel = np.ones((2,2), np.uint8)
    enhanced = cv2.dilate(binary, kernel, iterations=1)
    
    # Skala om till 28x28
    resized = cv2.resize(enhanced, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Centrera tecknet
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centered = np.zeros((28, 28), dtype=np.uint8)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        centered[(28-h)//2:(28-h)//2+h, (28-w)//2:(28-w)//2+w] = resized[y:y+h, x:x+w]
    else:
        centered = resized
    
    return centered.reshape(1, -1) / 255.0  # Normalisera

# Ladda upp en bild
uploaded_file = st.file_uploader("Ladda upp en bild", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    
    # Visa den förbehandlade bilden
    st.image(image, caption="Uppladdad bild", use_column_width=False)
    
    # Gör en förutsägelse
    prediction = model.predict(processed_image)[0]
    
    # Visa resultat
    st.success(f"Modellen förutspår att detta är en **{prediction}**!")