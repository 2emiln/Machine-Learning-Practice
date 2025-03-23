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

def preprocess_image(img, show_steps=False):
    """ Förbereder en ritad eller uppladdad bild så att den matchar MNIST-datasetets förprocessering och visar steg om checkboxen är aktiverad. """

    # Steg 1: Omvandla till gråskala
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    if show_steps:
        st.image(img, caption="Steg 1: Gråskalebild", width=150)

    # Steg 2: Invertera bilden
    img = cv2.bitwise_not(img)
    if show_steps:
        st.image(img, caption="Steg 2: Inverterad bild", width=150)

    # Steg 3: Binärisering
    threshold_value = threshold_otsu(img)
    binary = (img > threshold_value).astype(np.uint8) * 255
    if show_steps:
        st.image(binary, caption="Steg 3: Binäriserad bild", width=150)

    # Förbättring: Tjocka till siffran
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    if show_steps:
        st.image(binary, caption="Steg 3.5: Förstärkt siffra", width=150)

    # Steg 4: Bounding box
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = 4
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, binary.shape[1] - x)
        h = min(h + 2 * margin, binary.shape[0] - y)
    else:
        x, y, w, h = 0, 0, binary.shape[1], binary.shape[0]

    img_with_box = cv2.rectangle(binary.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2)
    if show_steps:
        st.image(img_with_box, caption="Steg 4: Bounding box identifierad", width=150)

    # Steg 5: Skala till 20x20
    max_side = max(w, h)
    scale = 20 / max_side
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    cropped_digit = binary[y:y+h, x:x+w]
    img_resized = cv2.resize(cropped_digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    img_square = np.zeros((20, 20), dtype=np.uint8)
    x_offset = (20 - new_w) // 2
    y_offset = (20 - new_h) // 2
    img_square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    if show_steps:
        st.image(img_square, caption="Steg 5: Skalad till 20x20", width=150)

    # Steg 6: Placera i 28x28
    img_padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    img_padded[y_offset:y_offset+20, x_offset:x_offset+20] = img_square
    if show_steps:
        st.image(img_padded, caption="Steg 6: Placerad i 28x28 bild", width=150)

    # Steg 7: Centrering
    cy, cx = center_of_mass(img_padded)
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img_final = cv2.warpAffine(img_padded, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if show_steps:
        st.image(img_final, caption="Steg 7: Centrerad bild", width=150)

    # Steg 8: Normalisering
    img_final = img_final.astype(np.float32) / 255.0  

    return img_final.flatten().reshape(1, -1)


def draw():
    """Sidan för att rita en siffra"""
    st.title("Ritningssidan")
    st.write("Här kan du rita en siffra (0-9).")

    # Checkbox för att visa preprocesseringsstegen
    show_steps = st.checkbox("Visa preprocesseringssteg")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = np.array(canvas_result.image_data, dtype=np.uint8)
        
        if st.button("🔍 Klassificera"):

            img_flattened = preprocess_image(img, show_steps)

            # Förutsäg siffra
            prediction = ml_model.predict(img_flattened)[0]
            st.success(f"🔮 Modellen tror att detta är en **{prediction}**!")


# ======= SIDOMENY ===========
page = st.sidebar.radio("Välj en sida:", ["Rita"])

if page == "Rita":
    draw()

