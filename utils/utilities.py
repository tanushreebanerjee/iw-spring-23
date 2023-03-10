from transformers import ViltProcessor, ViltForQuestionAnswering, YolosFeatureExtractor, YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import configs_py3 as configs
import matplotlib.pyplot as plt
import skimage.io as io
import torch
import os
import pandas as pd
import numpy as np
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
import torchvision
from torchvision.io import read_image

def load_dataset():
    df = pd.read_csv(configs.input_path)
    train_df = df[:configs.NUM_TRAIN]
    val_df = df[configs.NUM_TRAIN:configs.NUM_TRAIN + configs.NUM_VAL]
    test_df = df[configs.NUM_TRAIN + configs.NUM_VAL:configs.NUM_TRAIN + configs.NUM_VAL + configs.NUM_TEST]

    df = df.reset_index()  # make sure indexes pair with number of rows
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    test_df = test_df.reset_index()
    
    return df, train_df, val_df, test_df

def get_model_children(model):
    for idx, m in enumerate(model.named_modules()):
        print(idx, '->', m)
    return

def get_raw_val_prediction():
    
    model = ViltForQuestionAnswering.from_pretrained(configs.CHECKPOINT)
    processor = ViltProcessor.from_pretrained(configs.CHECKPOINT)

    df, train_df, val_df, test_df = load_dataset()

    
    original_model_outputs = []
 
    softmax_outputs = []

    for index, row in val_df.iterrows():
        imgId = row['image_id']
        imgFilename = 'COCO_' + configs.dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
        imgPath = configs.imgDir + imgFilename
        
        if not os.path.isfile(imgPath):
            continue
        
            
        image = Image.open(imgPath)
        
        
        # ground truth question for the first half of the dataset, and random question for the second half 
        if index < len(val_df)/2:
            text= row['original_question']
        else:
            text = row['random_question']
            
        
        # prepare inputs
        try:
            inputs = processor(image, text, return_tensors="pt")
        except Exception as e:
            I = io.imread(imgPath)
        
            plt.imshow(I)
            plt.axis('off')
            plt.show()
            print(e, "==> Skipping this image")
          
            original_model_outputs.append("N/A")
            softmax_outputs.append(0)
            continue                
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        softmax_output = torch.nn.functional.softmax(outputs.logits, dim=1)
        softmax_output = softmax_output[0, idx].item()
        softmax_outputs.append(softmax_output)
        
        original_model_outputs.append(model.config.id2label[idx])
  
    val_df[f'original_model_outputs'] = original_model_outputs
    val_df['softmax_outputs'] = softmax_outputs
    
    return val_df, original_model_outputs, softmax_outputs

def display_objects_in_image(img_path, obj_detection_checkpoint=configs.OBJECT_DETECTION_CHECKPOINT, obj_detection_threshold=0.9):
    feature_extractor = YolosFeatureExtractor.from_pretrained(obj_detection_checkpoint)
    model = YolosForObjectDetection.from_pretrained(obj_detection_checkpoint)
    processor = YolosImageProcessor.from_pretrained(obj_detection_checkpoint)
    
    image_tensor = read_image(img_path)
    image_pil = torchvision.transforms.ToPILImage()(image_tensor)
    
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    outputs = model(**inputs)
    
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > obj_detection_threshold
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=obj_detection_threshold)[0]
    labels = [model.config.id2label[int(label)] for label in results["labels"]]
    
    # draw bounding box and fill color
    annotated_image = draw_bounding_boxes(image_tensor, results["boxes"], labels=labels)
    
    # transform this image to PIL image
    annotated_image = torchvision.transforms.ToPILImage()(annotated_image)
    
    
    return annotated_image



def get_objects_in_image(img_path, obj_detection_checkpoint=configs.OBJECT_DETECTION_CHECKPOINT, obj_detection_threshold=0.9):
    feature_extractor = YolosFeatureExtractor.from_pretrained(obj_detection_checkpoint)
    model = YolosForObjectDetection.from_pretrained(obj_detection_checkpoint)
    processor = YolosImageProcessor.from_pretrained(obj_detection_checkpoint)
    
    image_tensor = read_image(img_path)
    image_pil = torchvision.transforms.ToPILImage()(image_tensor)
    
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    outputs = model(**inputs)
    
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > obj_detection_threshold
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=obj_detection_threshold)[0]
    labels = [model.config.id2label[int(label)] for label in results["labels"]]
    
    scores = [score.item() for score in results["scores"]]
    
    objects = dict(zip(labels, scores))
    
    return objects

