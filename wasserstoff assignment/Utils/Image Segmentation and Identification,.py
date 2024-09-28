#!/usr/bin/env python
# coding: utf-8

# In[58]:


#Image Segmentation and Identification


# In[65]:


# Import necessary libraries
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from google.colab import files


# In[68]:


# Upload the image file
uploaded = files.upload()


# In[70]:


# Get the uploaded image path (assuming only one file is uploaded)
img_path = list(uploaded.keys())[0]
print(f"Loading image: {img_path}")


# In[71]:


# Load and preprocess the image
img = Image.open(img_path).convert('RGB')
transform = T.ToTensor()
image = transform(img)


# In[72]:


# Load models
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model1 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model1.eval()


# In[73]:


# Object detection using Faster R-CNN
with torch.no_grad():
    prediction = model([image])

# Extract predictions
box, labels, scores = prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"]


# In[74]:


# Define COCO class names
coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
              "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
              "hair brush"]


# In[75]:


len(coco_names)


# In[76]:


# Display detected objects with bounding boxes
num = torch.argwhere(scores > 0.8).shape[0]
font = cv2.FONT_HERSHEY_SIMPLEX
img_read = cv2.imread(img_path)


# In[77]:


for i in range(num):
    x1, y1, x2, y2 = box[i].numpy().astype('int')
    class1 = coco_names[labels.numpy()[i] - 1]
    detected = cv2.rectangle(img_read, (x1, y1), (x2, y2), (0, 255, 0), 1)
    detected = cv2.putText(detected, class1, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

cv2_imshow(detected)


# In[79]:


# Object segmentation using Mask R-CNN
tensor = T.ToTensor()(img).unsqueeze(0)

with torch.no_grad():
    predictions = model1(tensor)

masks = predictions[0]['masks']
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']


# In[80]:


# Threshold for filtering out low-confidence predictions
threshold = 0.5


# In[81]:


# Convert the image to OpenCV format
image_np = np.array(img)
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

for i in range(len(masks)):
    if scores[i] > threshold:
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        label = labels[i].item()
        box = boxes[i].cpu().numpy().astype(int)

        # Create a color overlay for the mask
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        mask_colored = np.zeros_like(image_np, dtype=np.uint8)
        mask_colored[mask > 128] = color

        # Add mask to the image
        image_np = cv2.addWeighted(image_np, 1.0, mask_colored, 0.5, 0)

        # Draw the bounding box
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
        cv2.putText(image_np, coco_names[label - 1], (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2, cv2.LINE_AA)

cv2_imshow(image_np)

