#!/usr/bin/env python
# coding: utf-8

# In[11]:


#STEP 2: Object Extraction and Storage


# In[18]:


# Importing required libraries
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import os
import pandas as pd
import uuid
from google.colab import files


# In[19]:


# Upload image to Google Colab
uploaded = files.upload()

# Get the uploaded image path
img_path = next(iter(uploaded))  # Get the file name from uploaded

# Loading pre-trained Mask R-CNN model
model2 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model2.eval()


# In[14]:


# Define the transformation for the input image
transform = T.Compose([T.ToTensor()])


# In[15]:


# Defining all the file paths to variables
output = "segmented_objects"
object_file = "objects.csv"


# In[20]:


# Ensure the output directory exists
os.makedirs(output, exist_ok=True)


# In[21]:


# Loading and preprocessing the image
image = Image.open(img_path).convert("RGB")
tensor = transform(image)


# In[22]:


# Performing object detection
with torch.no_grad():
    predictions = model2([tensor])[0]

# Get the master ID for the image
master_id = str(uuid.uuid4())
objects = []


# In[23]:


# Iterating through the detected objects and saving each object as a separate image
for idx, mask in enumerate(predictions['masks']):
    mask = mask[0].mul(255).byte().cpu().numpy()

    # Generating unique ID for the object
    object_id = str(uuid.uuid4())

    mask_img = Image.fromarray(mask)
    segmented_img = Image.composite(image, Image.new("RGB", image.size), mask_img)
    output_path = os.path.join(output, f"object_{idx}.png")
    segmented_img.save(output_path)
    objects.append({
        "object_id": object_id,
        "master_id": master_id,
        "file_path": output_path
    })


# In[24]:


# Saving metadata to a CSV file
object_df = pd.DataFrame(objects)
object_df.to_csv(object_file, index=False)

print(f"Segmented objects saved in '{output}' and metadata saved in '{object_file}'")


# In[ ]:




