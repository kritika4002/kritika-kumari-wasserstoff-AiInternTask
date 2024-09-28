#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import numpy as np
from google.colab.patches import cv2_imshow


# In[2]:


# Load the Pascal VOC dataset 
voc_dataset = VOCSegmentation(root=".", year='2012', image_set='val', download=True)


# In[3]:


# Load the Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode


# In[4]:


# Process just one image from Pascal VOC dataset
image, target = voc_dataset[0]  # You can replace the index to process different images


# In[5]:


# Convert the PIL image to a tensor
image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension
# Print the shape of the image tensor
print("Image Tensor Shape:", image_tensor.shape)


# In[6]:


# Perform prediction
with torch.no_grad():
    predictions = model(image_tensor)

# Extract predictions
masks = predictions[0]['masks']
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Print predictions
print("Number of Predictions:", len(masks))
print("Masks Shape:", masks.shape)
print("Boxes:", boxes)
print("Labels:", labels)
print("Scores:", scores)


# In[7]:


# Set a confidence threshold
threshold = 0.5


# In[8]:


# Convert the image to a NumPy array for OpenCV
image_np = np.array(image)
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


# In[9]:


# Iterate through predictions and apply masks
for i in range(len(masks)):
    if scores[i] > threshold:
        print(f"Processing mask {i + 1}, Score: {scores[i]}")

        # Extract the mask, label, and box
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        label = labels[i].item()
        box = boxes[i].cpu().numpy().astype(int)

        print(f"Box Coordinates: {box}, Label: {label}")

        # Create a color overlay for the mask
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        mask_colored = np.zeros_like(image_np, dtype=np.uint8)
        mask_colored[mask > 128] = color

        # Add the mask to the image
        image_np = cv2.addWeighted(image_np, 1.0, mask_colored, 0.5, 0)

        # Draw the bounding box on the image
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
        cv2.putText(image_np, f"Label: {label}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

# Display the final image with masks and bounding boxes using cv2_imshow
cv2_imshow(image_np)


# In[10]:


# Save the output image
cv2.imwrite("voc_segmented_output.jpg", image_np)

print("Segmentation complete and image saved as voc_segmented_output.jpg")


# In[ ]:




