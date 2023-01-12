import streamlit as st

from joblib import load
from helpers.paths import Paths
from helpers.scale_and_predict import scale_and_predict

# Setup web service
st.title('Immobilien Preisvorhersage')
st.markdown('Geben sie die Eigenschaften der Immobilie ein und erhalten sie den vorhergesagten Preis')
st.header('Attribute der Immobilie')


# Load metadata and model
metadata = load(Paths.WEBSERVICE_META_DATA)
model = load(Paths.REGRESSOR_MODEL_DATA('GradientBoostingRegressor'))
features = metadata['asked_features']
options = metadata['options']

rows_per_column = len(features) // 2

col1, col2 = st.columns(2)
columns = ([col1] * (len(features) - rows_per_column)) + ([col2] * rows_per_column)

# Setup inputs
inputs = {}
for feature, column in zip(features, columns):
    with column:
        if feature in options.keys():
            inputs[feature] = st.selectbox(feature, options[feature])
        else:
            inputs[feature] = st.number_input(feature)

# Setup button
if st.button('Preis berechnen'):
    answer = scale_and_predict(model, metadata, inputs)
    st.success(f'Ihre Immobilie hat einen Wert von {"{:,.0f}".format(answer)} CHF.')
