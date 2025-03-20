import streamlit as st
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import center_of_mass
from skimage.filters import threshold_otsu

# Ladda ML-modellen
ml_model = joblib.load("models/best_model_RF.pkl")

def preprocess_image(img):
    """ Förbereder en ritad eller uppladdad bild så att den matchar MNIST-datasetets förprocessering och visar varje steg """

    # Steg 1: Om bilden har färgkanaler, konvertera till gråskala
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    st.image(img, caption="Steg 1: Gråskalebild", use_column_width=False, width=150)

    # Steg 2: Invertera bilden så att siffran blir svart på vit bakgrund
    img = cv2.bitwise_not(img)
    st.image(img, caption="Steg 2: Inverterad bild", use_column_width=False, width=150)

    # Steg 3: Binärisera bilden med Otsu's metod
    threshold_value = threshold_otsu(img)
    binary = (img > threshold_value).astype(np.uint8) * 255
    st.image(binary, caption="Steg 3: Binäriserad bild", use_column_width=False, width=150)

    # ** Förbättring: Tjocka till siffran innan bounding box detekteras **
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)  # Gör siffran tjockare
    st.image(binary, caption="Steg 3.5: Förstärkt siffra", use_column_width=False, width=150)

    # Steg 4: Hitta konturer och välj den största (mest sannolikt siffran)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Välj den största konturen (största siffran)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Hitta en rektangel runt konturen
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Lägg till en liten marginal runt bounding boxen
        margin = 4
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, binary.shape[1] - x)
        h = min(h + 2 * margin, binary.shape[0] - y)
    else:
        # Om inga konturer hittas, returnera en standard bounding box
        x, y, w, h = 0, 0, binary.shape[1], binary.shape[0]

    # Visa bounding box-bilden
    img_with_box = cv2.rectangle(binary.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2)
    st.image(img_with_box, caption="Steg 4: Bounding box identifierad", use_column_width=False, width=150)

    # Steg 5: Skala om till 20x20 samtidigt som aspect ratio bevaras
    max_side = max(w, h)
    scale = 20 / max_side

    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    # Klipp ut siffran och skala om (med bättre interpolering)
    cropped_digit = binary[y:y+h, x:x+w]
    img_resized = cv2.resize(cropped_digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # ** Lägg till padding för att få exakt 20x20 **
    img_square = np.zeros((20, 20), dtype=np.uint8)
    x_offset = (20 - new_w) // 2
    y_offset = (20 - new_h) // 2
    img_square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    st.image(img_square, caption="Steg 5: Skalad till 20x20", use_column_width=False, width=150)

    # Steg 6: Placera den i en 28x28 bild
    img_padded = np.zeros((28, 28), dtype=np.uint8)  # Viktigt att behålla uint8 här
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    img_padded[y_offset:y_offset+20, x_offset:x_offset+20] = img_square
    st.image(img_padded, caption="Steg 6: Placerad i 28x28 bild", use_column_width=False, width=150)

    # Steg 7: Centrera genom att använda Center of Mass
    cy, cx = center_of_mass(img_padded)
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img_final = cv2.warpAffine(img_padded, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    st.image(img_final, caption="Steg 7: Centrerad bild", use_column_width=False, width=150)

    # Steg 8: Normalisering (precis innan return)
    img_final = img_final.astype(np.float32) / 255.0  # Nu konverterar vi till float

    # Platta ut till en vektor (1x784) och returnera
    return img_final.flatten().reshape(1, -1)


def draw():
    """Sidan för att rita en siffra"""
    st.title("🖌️ Ritningssidan")
    st.write("Här kan du rita en siffra (0-9).")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = np.array(canvas_result.image_data, dtype=np.uint8)
        
        # Lägg till en knapp för att skicka bilden till modellen
        if st.button("🔍 Klassificera"):
            img_flattened = preprocess_image(img)

            # Förutsäg siffra
            prediction = ml_model.predict(img_flattened)[0]
            st.success(f"🔮 Modellen tror att detta är en **{prediction}**!")

def upload():
    """Sidan för att ladda upp en siffra"""
    st.title("📤 Uppladdningssidan")
    st.write("Ladda upp en bild på en siffra (0-9).")

    def preprocess_image(image):
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        enhanced = cv2.dilate(binary, kernel, iterations=1)

        resized = cv2.resize(enhanced, (28, 28), interpolation=cv2.INTER_AREA)

        contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centered = np.zeros((28, 28), dtype=np.uint8)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            centered[(28-h)//2:(28-h)//2+h, (28-w)//2:(28-w)//2+w] = resized[y:y+h, x:x+w]
        else:
            centered = resized

        return centered.reshape(1, -1) / 255.0, centered  

    uploaded_file = st.file_uploader("Ladda upp en bild", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Förprocessa uppladdad bild
        img_flattened = preprocess_image(img)

        # Förutsäg siffra
        prediction = ml_model.predict(img_flattened)[0]
        st.success(f"🔮 Modellen tror att detta är en **{prediction}**!")

# ======= SIDOMENY ===========
page = st.sidebar.radio("Välj en sida:", ["Rita", "Ladda upp"])

if page == "Rita":
    draw()
elif page == "Ladda upp":
    upload()
