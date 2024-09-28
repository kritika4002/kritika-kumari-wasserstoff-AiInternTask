#!/usr/bin/env python
# coding: utf-8

# In[82]:


#streamlit


# In[85]:


import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
import cv2


# In[84]:


get_ipython().system('pip install streamlit')


# In[88]:


# Set title
st.title("Object Extraction, Identification and Analysis")


# In[89]:


# Image upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# In[ ]:


if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Image successfully uploaded!")

    # User response section
    user_response = st.text_area("Submit your response here:")

    # Submit button
    if st.button("Submit"):
        if user_response:
            st.success("Your response has been submitted!")
            st.write("**Your Response:**", user_response)
        else:
            st.error("Please enter a response before submitting.")


# In[ ]:




