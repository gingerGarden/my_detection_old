import os, time, cv2, warnings
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from Module.utils.Convenience_Function import read_json

CODE_TEST = False


# Function
################################################################################
# seed 고정
def set_all_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
################################################################################



# 학습 환경 설정
################################################################################
# Seed 고정
BASIC_SEED = 1234
set_all_random_seed(seed_value = BASIC_SEED)

# UMP
"""
자동 혼합 정밀도(AMP)의 개념
- 혼합 정밀도: 서로 다른 데이터 타입(예: float32와 float16)을 연산에 사용하는 방식으로,
  메모리 사용량을 줄이고, GPU 성능을 개선할 수 있음.
- float16은 정밀도가 낮아 일부 연산에서는 부정확한 결과를 초래할 수 있음.

Autocast란?
- autocast는 컨텍스트 매니저로 주로 사용되며, `with torch.cuda.amp.autocast()` 블록
  내의 연산은 자동으로 낮은 정밀도로 수행됨.
- `autocast`는 연산에 가장 적합한 정밀도를 자동으로 선택하며, 대부분 flaot16을 사용하고,
  필요한 경우 float32로 되돌림.
- 장점
  1. 계산 성능 향상: 낮은 정밀도를 사용해, GPU의 처리 속도 향상 및 메모리 대역폭의 부하를
     줄일 수 있음.
  2. 메모리 절약
  3. 사용 용이성: 개발자가 직접 데이터 타입을 관리할 필요 없이
- 주의사항
  1. 정밀도와 정확도의 균형: 혼합 정밀도는 계산 성능을 향상시키지만, 최소한의 정확도 손실을
     감수해야할 수 있음. 대부분 이러한 손실을 무시할 수 있으나, 일부 경우에 문제가 될 수
     있음.
  2. 하드웨어 호환성: 모든 GPU가 float16 연산을 효율적으로 지원하지 않음.
- 일반적으로 `autocast` 사용 시, `torch.cuda.amp.GradScaler`를 함께 사용하여 그라이언트
  스케일링을 수행하는 것이 좋음. 이는 foat16에서 발생할 수 있는 언더플로우(underflow)나 
  오버플로우(overflow)문제 완화에 도움을 줌.
"""
USE_AMP = True
EARLY_STOPPING = True
if CODE_TEST:
    PATIENCE = 2
else:
    PATIENCE = 30
    
# Augment 설정
from Module.augmentation.aug import my_augmentation
AUG_Ins = my_augmentation(
    add_txt_p=0.2, reduce_size_p=0, flip_p=0.5,
    rs_param_dict={
        "num_range":range(1,4),
        "size_range":np.arange(0.3, 1.0, 0.05),
        "str_patn_list":[
            r"[A-Za-fh-ik-or-xz0-9]{1,2}",
            r"[A-Za-fh-ik-or-xz0-9]{1,3}[A-Za-fh-ik-or-xz0-9 _-]{0,3}[A-Za-fh-ik-or-xz0-9]{0,3}"
        ]
    },
    reduce_max=1.2,
    flip_key_list=["horizontal", "vertical", "all"]
)
################################################################################



# Data seperate
################################################################################
DATASET_STRING = "Dataset"
if CODE_TEST:
    K_SIZE = 1
else:
    K_SIZE = 5

SEED_LIST = [1111, 2222, 3333, 4444, 5555]
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1

SEPERATE_KEY_COLUMN_LIST = ["directory", "txt_json"]
EXTERNAL = False
EXTERNAL_COLUMN = "directory"
EXTERNAL_ELEMENT = "E"
################################################################################



# Path
################################################################################
# Source Data의 경로
SOURCE_PATH = "../SourceData"
# image들이 들어가 있는 디렉터리 경로
IMG_DIR = "Image"
IMG_PARENTS_PATH = f"{SOURCE_PATH}/{IMG_DIR}"
# txt에 대한 annotation들이 들어가 있는 디렉터리 경로
TXT_JSON_DIR = "TXT_Json"
TXT_JSON_PARENTS_PATH = f"{SOURCE_PATH}/{TXT_JSON_DIR}"
# txt에 대한 annotation 결과를 보기 좋게 정리해놓은 경로
BASIC_KEY_DF_NAME = "txt_annotation_key_df.csv"
_basic_key_df = pd.read_csv(f"{SOURCE_PATH}/{BASIC_KEY_DF_NAME}")
if CODE_TEST:
    BASIC_KEY_DF = _basic_key_df[:1000]
else:
    BASIC_KEY_DF = _basic_key_df
    
# 이미지와 annotation들을 하나의 json 파일로 정리해놓은 학습에 사용되는 키 데이터
JSON_FILE = "my_torch_style_txt_annotation.json"
LABEL_DICT_PATH = f"{SOURCE_PATH}/{JSON_FILE}"
LABEL_DICT = read_json(LABEL_DICT_PATH)
# 무작위로 분할된 k개의 Dataset에 대한 index dictionary의 경로
INDEX_DICT_FILE = "Index_dictionary.pickle"
INDEX_DICT_PATH = f"{SOURCE_PATH}/{INDEX_DICT_FILE}"

# Process 전반 편의 기능
## Log
LOG_DIR = "Log"
ITERATION_LOG_DIR = "Iter_Log"
ITER_LOG_KEY = f"{LOG_DIR}/{ITERATION_LOG_DIR}/model_training_log"
TRAINING_LOG_DIR = "Train_Log"
## Model
MODEL_DIR = "Model"
################################################################################



# Model hyper parameter
################################################################################
BATCH_SIZE = 8
WORKER_NUM = 4
NUMBER_OF_CLASS = 2

EPOCHS = 200
"""
Gradient clipping은 역전파(backpropagation) 과정에서 기울기(gradient)의 크기를 제한하여
안정성을 개선하는 데 도움을 주는 기법임.
`max_norm`은 gradient clipping 수행 시, 기울기 벡터의 최대 길이(norm)를 정의함.
 1. 기울기 폭발(Gradient Exploding) 방지: 신경망 훈련 시, 기울기가 매우 큰 값으로 폭발할 수 있음.
    특히 깊은 네트워크에서 흔히 발생하는 문제로, 기울기가 너무 크면 학습 과정에서 가중치가 불안정하게
    되어, 모델의 성능 저하가 발생할 수 있음.
 2. Max Norm의 역할: `max_norm`은 기울기 벡터의 최대 길이로, 역전파 동안 계산된 기울기의 크기가
    `max_norm` 값보다 큰 경우, 그 크기를 `max_norm`로 줄여서, 기울기의 크기가 특정 임계값을
    초과하지 않게 함.
 3. 기울기 벡터의 조정: Gradient clipping 수행 시, 기울기 벡터는 그 방향은 유지하되 크기만 조정됨.
    이는 기울기의 방향은 모델이 최적화해야 할 방향을 나타내지만, 그 크기가 너무 큰 경우 문제의 원인이
    될 수 있음.
 4. 훈련 안정성 개선: Gradient clipping은 특히 장기 의존성을 다루는 RNN(Recurrent Neural Network)과
    같은 네트워크에서 효과적임. 이 기법을 사용함으로써 학습 과정을 더 안정적으로 만들고, 모델의 수렴을
    개선할 수 있음.
    
# max norm의 가장 좋은 값에 대한 문헌 조사 실시하자.
# 일반적으로 1~10사이로 설정한다고 하나, 이는 권장 사항이다.
"""
GRADIENT_CLIIPING_MAX_NORM = 5
################################################################################