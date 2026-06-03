import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib

# Ladda bästa modellen (KNN om den vann)
model = joblib.load("mnist_model.pkl")

st.title("MNIST Digit Classifier – KNN-version")
st.write("Rita en siffra (0–9) i rutan nedan så försöker modellen gissa vad du ritat.")

# Rensa canvas
if st.button("Rensa canvas"):
    if "canvas" in st.session_state:
        del st.session_state["canvas"]
    st.rerun()

# Canvas-inställningar
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# När användaren ritat något
if canvas_result.image_data is not None:

    img = canvas_result.image_data

    # Kontroll: är bilden tom?
    if np.mean(img) > 250:
        st.write("Rita en siffra i rutan.")
        st.stop()

    # Gör om till PIL-bild i gråskala
    pil_img = Image.fromarray(img.astype("uint8")).convert("L")

    # Resize grovt till 28x28
    pil_img = pil_img.resize((28, 28))
    arr = np.array(pil_img)

    # Invertera (vitt blir 0, siffran blir höga värden)
    arr = 255 - arr

    # Binarisera (gör siffran tydligare)
    arr_bin = (arr > 50).astype(np.uint8) * 255

    # Hitta bounding box runt siffran
    rows = np.where(arr_bin.max(axis=1) > 0)[0]
    cols = np.where(arr_bin.max(axis=0) > 0)[0]

    if len(rows) == 0 or len(cols) == 0:
        st.write("Kunde inte hitta någon siffra, försök rita tydligare.")
        st.stop()

    r_min, r_max = rows[0], rows[-1]
    c_min, c_max = cols[0], cols[-1]

    # Cropa till siffran
    digit = arr[r_min:r_max+1, c_min:c_max+1]

    # Resize till 20x20
    digit_img = Image.fromarray(digit.astype("uint8")).resize((20, 20))
    digit_arr = np.array(digit_img)

    # Pad till 28x28 (centrera siffran)
    padded = np.zeros((28, 28), dtype=np.uint8)
    start_row = (28 - 20) // 2
    start_col = (28 - 20) // 2
    padded[start_row:start_row+20, start_col:start_col+20] = digit_arr

    # Flatten
    img_array = padded.reshape(1, -1)

    # Prediktion
    prediction = model.predict(img_array)[0]

    st.write(f"### Modellens gissning: **{prediction}**")

    st.write("Förbehandlad bild (28x28):")
    st.image(padded, width=150)
