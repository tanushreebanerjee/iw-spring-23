from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import YolosFeatureExtractor, YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import configs_py3 as configs
import matplotlib.pyplot as plt
import skimage.io as io
import torch
import os
import pandas as pd
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.io import read_image
from nltk.corpus import wordnet as wn
import nltk
from transformers import BlipProcessor, BlipForConditionalGeneration

def get_preds(df, object_annotations=False):
    # get outputs with softmax scores
    # first half of val set with original questions
    # second half with random questions
    

    df_original, _, _ = get_raw_vqa_output(
        df=df[:len(df)//2], 
        random_question=False,
        object_annotations=object_annotations)

    df_random, _, _ = get_raw_vqa_output(
        df=df[len(df)//2:],
        random_question=True,
        object_annotations=object_annotations)
    #combine both dfs
    df = pd.concat([df_original, df_random], ignore_index=True)
    return df

def load_dataset(split='all', 
                 num_train=configs.NUM_TRAIN, 
                 num_val=configs.NUM_VAL, 
                 num_test=configs.NUM_TEST, 
                 input_path=configs.input_path):
    
    
    df = pd.read_csv(input_path)
    train_df = df[:num_train].reset_index()
    val_df = df[num_train:num_train + num_val].reset_index()
    test_df = df[num_train + num_val:num_train + num_val + num_test].reset_index()
    
    if split=='full':
        return df   
        
    elif split == "train":
        return train_df
    
    elif split == "val":
        return val_df
    
    elif split == "test":
        return test_df
    
    if split == "all":
        return df, train_df, val_df, test_df
    
    else:
        raise Exception("Invalid split argument. Must be one of 'full', 'train', 'val', 'test', or 'all'.")

def prepare_inputs(df, random_question=False):
    
    image_paths = []
    question_inputs = []

    
    for _, row in df.iterrows():
        
        imgId = row['image_id']
        imgFilename = 'COCO_' + configs.dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
        imgPath = configs.imgDir + imgFilename
        
        if not os.path.isfile(imgPath):
            continue
        
        if random_question:
            text = row['random_question']
        else:
            text = row['original_question']
            
        image_paths.append(imgPath)
        question_inputs.append(text)
  
    return image_paths, question_inputs

def get_raw_vqa_output(model=ViltForQuestionAnswering.from_pretrained(configs.CHECKPOINT), 
                       processor=ViltProcessor.from_pretrained(configs.CHECKPOINT), 
                       feature_extractor=YolosFeatureExtractor.from_pretrained(configs.OBJECT_DETECTION_CHECKPOINT),
                       df=load_dataset(split="val"), 
                       random_question=False,
                       object_annotations=False):
    
    original_model_outputs = []
 
    softmax_outputs = []
    
    image_paths, question_inputs = prepare_inputs(df, random_question)
    

    for imgPath, text in zip(image_paths, question_inputs):
        
        
        if object_annotations:
            results = run_object_detection(imgPath, model=model, feature_extractor=feature_extractor, processor=processor)
            image = annotate_objects_in_image(imgPath, results, model=model)
        else:
            image = Image.open(imgPath)
        
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
  
    df['original_model_outputs'] = original_model_outputs
    df['softmax_outputs'] = softmax_outputs
    
    return df, original_model_outputs, softmax_outputs

def run_object_detection(img_path, 
                         model=YolosForObjectDetection.from_pretrained(configs.OBJECT_DETECTION_CHECKPOINT), 
                         feature_extractor=YolosFeatureExtractor.from_pretrained(configs.OBJECT_DETECTION_CHECKPOINT),
                         processor=YolosImageProcessor.from_pretrained(configs.OBJECT_DETECTION_CHECKPOINT),
                         threshold=0.9):
    
    image_tensor = read_image(img_path)
    image_pil = torchvision.transforms.ToPILImage()(image_tensor)
    
    try:
        inputs = feature_extractor(images=image_pil, return_tensors="pt")
    except Exception as e:
        I = io.imread(img_path)
        
        plt.imshow(I)
        plt.axis('off')
        plt.show()
        print(e, "==> Skipping this image")
        return None

    
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    return results

def annotate_objects_in_image(img_path, 
                              results,
                              model=YolosForObjectDetection.from_pretrained(configs.OBJECT_DETECTION_CHECKPOINT),
                              bbox_width=3):
    
    image_tensor = read_image(img_path)
    
    if results == None:
        return torchvision.transforms.ToPILImage()(image_tensor)
    
    
    labels = [model.config.id2label[int(label)] for label in results["labels"]]
    
    annotated_image = draw_bounding_boxes(image_tensor, results["boxes"], labels=labels, width=bbox_width)
    annotated_image = torchvision.transforms.ToPILImage()(annotated_image)
    
    
    return annotated_image

def get_objects_in_image(results,
                         model=YolosForObjectDetection.from_pretrained(configs.OBJECT_DETECTION_CHECKPOINT)):
    
    if results == None:
        return None
    
    labels = [model.config.id2label[int(label)] for label in results["labels"]]
    
    scores = [score.item() for score in results["scores"]]
    
    objects_in_image = dict(zip(labels, scores))
    
    return objects_in_image

def get_thresholded_output(original_model_outputs, softmax_outputs, threshold=0.5):
    thresholded_model_outputs = []
    for idx in range(len(softmax_outputs)):
        
        if softmax_outputs[idx] < threshold:
            thresholded_model_outputs.append("ABSTAIN")
        else:
            thresholded_model_outputs.append(original_model_outputs[idx])
    return thresholded_model_outputs

def get_image_caption(image_path, 
                      model=BlipForConditionalGeneration.from_pretrained(configs.IMAGE_CAPTIONING_CHECKPOINT), 
                      processor=BlipProcessor.from_pretrained(configs.IMAGE_CAPTIONING_CHECKPOINT),
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    model = model.to(device)
    raw_image = Image.open(image_path).convert('RGB')

    # conditional image captioning
    text = ""
    inputs = processor(raw_image, text, return_tensors="pt").to(device)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def get_synonyms(word):
    synonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    synonyms = [x.replace("_", " ") for x in synonyms]
    return list(set(synonyms))


def is_similar(obj1_str, obj2_str, threshold=configs.OBJECT_SIMILARITY_THRESHOLD):
    similarity = 0
    obj1_synsets = wn.synsets(obj1_str, pos=wn.NOUN)
    obj2_synsets = wn.synsets(obj2_str, pos=wn.NOUN)
    for obj1_synset in obj1_synsets:
        for obj2_synset in obj2_synsets:
            similarity = max(similarity, obj1_synset.path_similarity(obj2_synset))
    
    return similarity > threshold, similarity

def is_question_related(question, objects_in_image, threshold=configs.OBJECT_SIMILARITY_THRESHOLD):
    max_similarity_score = 0
    
    is_similar_final = False
    
    most_similar_word_pair = (None, None)
    
    if len(question) == 0 or len(objects_in_image) == 0:
        return True, max_similarity_score, most_similar_word_pair
    
    question = question.split()
    
    for word in question:
        for obj in objects_in_image:
            is_similar_bool, similarity_score = is_similar(word, obj, threshold=threshold)
            max_similarity_score = max(max_similarity_score, similarity_score)
            is_similar_final = is_similar_final or is_similar_bool
            most_similar_word_pair = (word, obj) if similarity_score == max_similarity_score  else most_similar_word_pair
            
    if is_similar_final:
        return is_similar_final, max_similarity_score, most_similar_word_pair
    
    return is_similar_final, max_similarity_score, most_similar_word_pair

def get_similarities(df, mode, threshold=configs.OBJECT_SIMILARITY_THRESHOLD):
    is_similar = []
    similarity_score = []
    for index, row in df.iterrows():
        question = row[f'{mode}_question']
        if row.objects_in_image == None:
            objects_in_image = []
        elif type(row.objects_in_image) == str:
            objects_in_image = eval(row.objects_in_image)
            
        
        if type(row.objects_in_image) == dict:
            objects_in_image = list(row.objects_in_image.keys())
        
        is_similar_final, max_similarity_score, most_similar_word_pair = is_question_related(question, 
                                                                                                   objects_in_image, 
                                                                                                   threshold=threshold)
        
        is_similar.append(is_similar_final)
        similarity_score.append(max_similarity_score)


    df[f"{mode}_is_similar_{round(threshold, 1)}"] = is_similar
    df[f"{mode}_similarity_score"] = similarity_score
    
    return df