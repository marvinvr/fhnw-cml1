import joblib
import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Immobilien Preisvorhersage')
st.markdown('Geben sie die Eigenschaften der Immobilie ein und erhalten sie den vorhergesagten Preis')

st.header("Plant Features")
col1, col2 = st.columns(2)
with col1:
    plot_area = st.number_input('Plot area (m^2)', 1, 1000, 1)
    living_space = st.number_input('Living space (m^2)', 1, 1000, 1)
with col2:
    zip = st.number_input('Zip', 1, 99999, 1)

#make the prediction if the button is clicked
if st.button("Berechne den vorhergesagten Preis"):
    result = predict(np.array([[plot_area, living_space, zip]]))
    st.text(result[0])
