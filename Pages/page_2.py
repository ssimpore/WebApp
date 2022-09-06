import streamlit as st

st.title("Welcome to the page 2")

selection = st.date_input('Date du jour')

st.write('The date is : ', selection)
