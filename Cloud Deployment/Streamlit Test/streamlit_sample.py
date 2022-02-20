import streamlit as st
import matplotlib.pyplot as plt
import cv2
import os

# py -m streamlit run streamlit_sample.py to deploy (Windows) or you can remove the py -m if you set env var to [PYTHON_PATH]/Scripts

# Image path
path = os.environ["DATA_PATH"] + "/images/animals/dog.jpg"

st.title("Web app using Streamlit")

# Display image
fig = plt.figure(figsize=(3,3))
plt.imshow(cv2.imread(path)[:,:,::-1])
plt.axis("off")
st.pyplot(dpi=300,fig=fig,clear_figure=True)
