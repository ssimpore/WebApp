import streamlit as st

st.title("Welcome to the page 3")

selection = st.slider('Selection de date: ', 0, 100, 10)

st.write('The selection is: ', selection)
