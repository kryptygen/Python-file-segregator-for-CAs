import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os

st.title("File Segregator App")

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

if st.button("Select Folder"):
    folder = select_folder()
    if folder:
        st.session_state.selected_folder = folder

if "selected_folder" in st.session_state:
    st.write("Selected:", st.session_state.selected_folder)

    if st.button("Process Folder"):

        from main_v2 import main_process

        with st.spinner("Processing files..."):
            result = main_process(st.session_state.selected_folder)

        if result["status"] == "success":
            st.success(result["message"])
        else:
            st.error(result["message"])
