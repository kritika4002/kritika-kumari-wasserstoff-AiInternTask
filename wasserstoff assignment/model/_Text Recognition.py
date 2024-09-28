#!/usr/bin/env python
# coding: utf-8

# In[43]:


##STEP 4 --> Text Recognition


# In[55]:


import pytesseract
from PIL import Image
from google.colab import files


# In[50]:


get_ipython().system('sudo apt update')
get_ipython().system('sudo apt install tesseract-ocr')
get_ipython().system('pip install pytesseract')


# In[56]:


# Update the Tesseract command path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# In[57]:


# Step 1: Upload an image
uploaded = files.upload()


# Load the uploaded image
for file_name in uploaded.keys():
    image = Image.open(file_name)

# Recognize text from the image
text = pytesseract.image_to_string(image)

# Print the recognized text
print("Recognized Text:")
print(text)


# In[ ]:




