import os
import streamlit as st
import tempfile
import helper as help


st.set_page_config(page_title="new doc")

st.sidebar.header("Query a new document")


uploaded_file = st.file_uploader("Choose a new file to query...")
if uploaded_file:
    st.write(uploaded_file.name)
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    chain = help.generate_db(path, uploaded_file.name)
    question = st.text_input('Enter query here')
    if question:
        answer, source = help.invoke(chain, question)
        st.write(answer)
        st.write(source)