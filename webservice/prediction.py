import joblib
import streamlit as st
import pandas as pd
import numpy as nps

def predict(data):
    reg = joblib.load("einfaches_modell.sav")
    return reg.predict(data)
