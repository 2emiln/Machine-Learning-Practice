import streamlit as st
import numpy as np
import joblib
import cv2
from streamlit_drawable_canvas import st_canvas

# Ladda modellen
ml_model = joblib.load("models/best_model_RF.pkl")

def preprocess_and_center_image(img):
    """ Gör ritade bilder lika uppladdade bilder genom att invertera färger. """
    
    # 🔥 1. Omvandla RGBA till RGB (viktigt för ritade siffror från st_canvas)
    if img.shape[-1] == 4:  # Kontrollera om det finns en alfakanal (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # 🔥 2. Konvertera till gråskala
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  

    # 🔥 3. Förbättra kontrasten med binär tröskel
    _, img_binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

    # 🔥 4. **Invertera bilden så att siffran blir svart på vit bakgrund**
    img_binary = cv2.bitwise_not(img_binary)

    # 🔥 5. Förstärk siffran med dilation (gör tunna streck tjockare)
    kernel = np.ones((5,5), np.uint8)  
    img_enhanced = cv2.dilate(img_binary, kernel, iterations=2)

    # 🔥 6. Hitta bounding box för siffran
    coords = cv2.findNonZero(img_enhanced)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # 🔥 7. Lägg till padding så att vi inte klipper siffran
        padding = 20  
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, img_enhanced.shape[1] - x)
        h = min(h + 2 * padding, img_enhanced.shape[0] - y)

        img_crop = img_enhanced[y:y+h, x:x+w]
    else:
        img_crop = np.zeros((28, 28), dtype=np.uint8)

    # 🔥 8. Se till att siffran blir tillräckligt stor i rutan
    size = max(w, h, 24)  
    centered_img = np.ones((size, size), dtype=np.uint8) * 255  # **Ändrat från 0 till 255**

    # 🔥 9. Säkerställ att siffran är i mitten
    if coords is not None:
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        centered_img[y_offset:y_offset+h, x_offset:x_offset+w] = img_crop

    # 🔥 10. Skala om med en bättre interpolationsmetod
    img_resized = cv2.resize(centered_img, (28, 28), interpolation=cv2.INTER_NEAREST)

    # 🔥 11. Extra binär tröskel efter skalning för skarpa kanter
    _, img_final = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)

    # 🔥 12. Normalisering (se till att vi får exakta 0 och 1)
    img_normalized = img_final / 255.0
    img_normalized = np.clip(img_normalized, 0, 1)

    # 🔥 13. Flattena för Scikit-Learn
    img_flattened = img_normalized.flatten().reshape(1, -1)

    return img_flattened, img_final







def draw():
    st.title("Ritningssidan")
    st.write("Här kan du rita en siffra (0-9).")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = np.array(canvas_result.image_data, dtype=np.uint8)

        # Förbättra, centrera och normalisera bilden
        img_flattened, img_resized = preprocess_and_center_image(img)

        # Debug: Se max och min värden i bilden
        st.write(f"Max pixelvärde: {np.max(img_resized)}")
        st.write(f"Min pixelvärde: {np.min(img_resized)}")

        # Debug: Visa bilden innan den skickas till modellen
        st.image(img_resized, caption="Bild efter förbehandling", width=150)

        # Visa den bearbetade bilden
        st.image(img_resized, caption="Centrerad och förbehandlad bild (28x28)", width=150)

        st.write("🔥 Input till modellen (vektoriserad):")
        st.write(img_flattened)
        st.write(f"Maxvärde i vektor: {np.max(img_flattened)}")
        st.write(f"Minvärde i vektor: {np.min(img_flattened)}")

        # Gör en förutsägelse
        prediction = ml_model.predict(img_flattened)[0]

        st.write(f"Modellen tror att detta är en **{prediction}**!")

def upload():
    st.title("Uppladdningssidan")
    st.write("Här kan du ladda upp en bild på en siffra (0-9).")
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

# Skapa sidomeny för navigation
page = st.sidebar.radio("Välj en sida:", ["Ritning", "Uppladdning"])

if page == "Ritning":
    draw()
if page == "Uppladdning":
    upload()