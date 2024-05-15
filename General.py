import streamlit as st
import helper as help

st.set_page_config(
    page_title="Hello"
)

st.write("# Welcome! This program can be used to query any PDF file of your\
         choice. Select a new document to start.")
st.sidebar.success("Select an option above.")