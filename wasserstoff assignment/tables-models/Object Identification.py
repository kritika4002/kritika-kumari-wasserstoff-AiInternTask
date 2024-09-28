#!/usr/bin/env python
# coding: utf-8

# In[25]:


#STEP 3 --> Object Identification


# In[26]:


# Rest of your object identification code
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow


# In[27]:


# Load the pre-trained Faster R-CNN model for detection
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


# In[28]:


from google.colab import files
from PIL import Image

#  Upload the image
uploaded = files.upload()

#  Load and convert the image to RGB
for filename in uploaded.keys():
    img_path = f"/content/{filename}"  # Use the uploaded filename
    print(f"Loading image: {img_path}")
    img = Image.open(img_path).convert('RGB')


# In[29]:


# Define the object detection model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


# In[30]:


#Apply transformations
transform = T.ToTensor()
image = transform(img)


# In[31]:


# Set the model to evaluation mode
model.eval()

# Perform object detection
with torch.no_grad():
    prediction = model([image])


# In[36]:


# Extract bounding boxes, labels, and confidence scores
boxes = prediction[0]["boxes"]
labels = prediction[0]["labels"]
scores = prediction[0]["scores"]

#  Load the PASCAL VOC names
coco_names = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# In[37]:


#  Define a confidence threshold
threshold = 0.8


# In[38]:


# Convert the image for OpenCV display
import numpy as np
img_cv = np.array(img)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)


# In[40]:


#  Draw bounding boxes and labels
print("Drawing bounding boxes and labels...")
for i in range(len(boxes)):
    if scores[i] > threshold:
        x1, y1, x2, y2 = boxes[i].numpy().astype('int')  # Extract box coordinates
        label_id = labels[i].item()  # Extract label ID

        # Check if the label_id is within the range of pascal_voc_names
        if label_id < len(coco_names):
            class_name = coco_names[label_id]  # Map label ID to class name
        else:
            class_name = "Unknown"


        print(f"Object {i + 1}: {class_name}, Box: {x1}, {y1}, {x2}, {y2}")

        # Draw the bounding box and label on the image
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with bounding boxes and labels
cv2_imshow(img_cv)

# Save the output image (optional)
cv2.imwrite("output_with_bboxes.jpg", img_cv)


# In[ ]:




