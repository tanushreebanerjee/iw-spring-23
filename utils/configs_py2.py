# -*- coding: future_fstrings -*-
from datetime import date

today = date.today()

EXPERIMENT_NAME = f"{today}_abstain_softmax_thresholding"
CHECKPOINT = "dandelin/vilt-b32-finetuned-vqa" # ViLT for VQA
ROOT = f".."
RANDOM_SEED = 42

NUM_TRAIN = 1000
NUM_VAL = 1000
NUM_TEST = 1000

dataDir		=f'{ROOT}/VQA'

versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType ='val2014' # MY ADDITION: changed from train2014 to val2014

annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
quesTypesFile = "%s/QuestionTypes/%s_question_types.txt" % (dataDir, dataType)
resFile = '%s/Results/%s%s_%s_%s_results.json' %(dataDir, versionType, taskType, dataType, dataSubType)

experimentDatasetDir = f'{ROOT}/data/{EXPERIMENT_NAME}'
dataset_path = "%s/%s%s_%s_%s.csv" % (experimentDatasetDir, versionType, taskType, dataType, dataSubType)
