import streamlit as st

### To test run the app on localhost
# $ pip install streamlit
# $ streamlit run streamlit_app.py

#-------------------------------- Introduction --------------------------------#

st.write("""
# HDB Resale Prices Predictor
Hello *World*!
"""
)

#----------------------------------- Sidebar ----------------------------------#

placeholder = st.sidebar.checkbox(
    label="Hi I'm a checkbox",
    value=False,
)