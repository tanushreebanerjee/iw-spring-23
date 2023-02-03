#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datasets import load_dataset, load_dataset_builder, Image, get_dataset_split_names
import pandas as pd


# In[3]:


DATASET_NAME = "HuggingFaceM4/VQAv2"

ROOT = f".."
DATASET_PATH = f"{ROOT}/datasets/VQAv2.csv"


# In[4]:


ds_builder = load_dataset_builder(DATASET_NAME)
ds_description = ds_builder.info.description
ds_features = ds_builder.info.features
dataset = load_dataset(DATASET_NAME, streaming=True)
ds_split_names = get_dataset_split_names(DATASET_NAME)


# In[5]:


question_type = []
multiple_choice_answer = []
answers = []
image_id = []
answer_type = []
question_id = []
question = []
image = []
split = []


# In[11]:


for split_name in ["test"]:#ds_split_names:
    print(split_name)
    for example in dataset[split_name].take(1000):
        question_type.append(example["question_type"])
        multiple_choice_answer.append(example["multiple_choice_answer"])
        answers.append(example["answers"])
        image_id.append(example["image_id"])
        answer_type.append(example["answer_type"])
        question_id.append(example["question_id"])
        question.append(example["question"])
        image.append(example["image"])
        split.append(split_name)
    
    
        



# In[ ]:


df_dict = {"question_type": question_type,
           "multiple_choice_answer": multiple_choice_answer,
           "answers": answers,
           "image_id": image_id,
           "answer_type": answer_type,
           "question_id": question_id,
           "question": question,
           "image":image,
           "split": split}

df = pd.DataFrame.from_dict(df_dict)
df.to_csv(DATASET_PATH)
