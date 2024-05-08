import datetime, time, pickle, os, shutil, sys, json, re
import numpy as np
import pandas as pd
from typing import Union




# 순수 편의 기능
#####################################################################################
# 소모 시간 측정
def time_checker(start:float) -> str:
    """
    start(float - time.time()) 부터 time_checker() 코드 실행까지 걸린 시간 출력
    '0:01:55.60'
    """
    # 소모 시간 측정
    end = time.time()
    second_delta = (end - start)
    result = decimal_seconds_to_time_string(decimal_s=second_delta)
    
    return result


def decimal_seconds_to_time_string(decimal_s):
    time_delta = datetime.timedelta(seconds=decimal_s)
    str_time_delta = str(time_delta).split(".")
    time1 = str_time_delta[0]
    if len(str_time_delta) == 1:
        time2 = "00"
    else:
        time2 = str_time_delta[1][:2]
    return f"{time1}.{time2}"



def new_dir_maker(dir_path:str, makes_new=True):
    """
    디렉터리 존재 유무 확인 후, 새로운 디렉터리 생성
    -------------------------------------------------------
    >>> 디렉터리 존재 시 이를 삭제하고 새로운 디렉터리 생성
    >>> makes_new가 True일 때만 삭제하고 새로운 디렉터리 생성
    """
    if os.path.exists(dir_path):
        
        if makes_new:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            pass
    else:
        os.mkdir(dir_path)

    
    
# Process를 필요한만큼 쪼갠다.
def Split_process_target(directory_list: list, process_size: int, process_number: int)->list:
    """
    directory_list를 process의 갯수(process_size)만큼 쪼개고, 해당 proces의 번호(process_number)에
    해당하는 directory_list만을 가지고 온다.
    >>> process를 쪼개서 별도로 실행할 때, 그 process에 해당하는 Data를 정의.
    """
    # 한 Process내 데이터의 크기
    total_length = len(directory_list)
    split_size = int(np.ceil(total_length/process_size))

    # indexing
    start_index = split_size * process_number
    end_index = split_size * (process_number + 1)
    
    result = directory_list[start_index:end_index]
    
    return result
#####################################################################################



            
# 특정 확장자 읽고 쓰기
#####################################################################################      
def save_pickle(data, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
        
        
def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
    
    
    
def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)



def write_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f)
        
        
        
def read_lineTxTFile_to_DataFrame(txtFilePath):
    """
    txt 파일의 각 line을 각각의 Record로 하여 DataFrame으로 가지고 온다.
    """
    f = open(txtFilePath, 'r')
    lines = f.readlines()

    stack_list = []
    for line in lines:
        stack_list.append(line)
    f.close()

    txt_df = pd.DataFrame({"lines":stack_list})
    # 맨 끝에 있는 \n 제거
    txt_df["lines"] = txt_df["lines"].str.replace("\n$", "", regex=True)
    
    return txt_df
#####################################################################################






# DataFrame 조작
#####################################################################################
# Excel로부터 DataFrame을 가지고 온다.
def get_DataFrame_from_excel(excel_path):
    
    df = pd.read_excel(excel_path, dtype='str')
    
    # 대상 column의 이름들만 가지고 온다.
    true_column_list = []
    for targetCol in df.columns:

        if len(re.findall("Unnamed: [0-9]+", targetCol)) != 1:
            true_column_list.append(targetCol)
            
    return df[true_column_list]



# DataFrame을 seed에 맞게 무작위로 섞음.
def shuffle_DataFrame(df, seed=None):
    """
    DataFrame을 seed에 맞게 무작위로 섞는다.
    """
    if seed is not None:
        np.random.seed(seed)
    index_arr = np.array(df.index)
    np.random.shuffle(index_arr)
    result = df.loc[index_arr].reset_index(drop=True)
    
    return result
#####################################################################################





# 자주 쓰는 통계지표
#####################################################################################
from tabulate import tabulate
# DataFrame을 Markdown으로 print 한다.
def show_markdown_df(df):

    pd.options.display.max_columns = None
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=True))
    
    
    
# Numpy Array로부터 pandas DataFrame 형태의 빈도표를 출력한다.    
def freq_table_by_np_array(np_array, verbose=False):
    """
    np.array를 기반으로 빈도표 생성
    >>> class, n, % 로 이루어진 빈도표 출력.
    >>> markdown으로 print 가능
    """
    # Class, 빈도를 추출한다.
    index_arr, value_arr = np.unique(np_array, return_counts=True)

    # 빈도표를 생성한다. 
    freq_table = pd.DataFrame({"class":index_arr, "n":value_arr})

    # %를 계산한다.
    freq_table["%"] = np.round((freq_table["n"]/sum(freq_table["n"]))*100, 1)

    margin_df = pd.DataFrame({"class":["total"], "n":[sum(freq_table["n"])], "%":[sum(freq_table["%"])]})
    freq_table = freq_table.append(margin_df)
    freq_table.reset_index(drop=True, inplace=True)
    
    if verbose:
        show_markdown_df(freq_table)
        
    return freq_table



class make_pretty_freq_table:
    """
    DataFrame에 있는 column들(column_list)에 대한 빈도표를 산출한다.
    -------------------------------------------------------------------
    1. 빈도표 생성 - index로 정렬, Series를 DataFrame으로 변환
    2. ratio 컬럼 추가
    3. total record 추가
    """
    def __init__(self, df:pd.core.frame.DataFrame, column_list:list, total_record=False):
        self.df = df
        self.column_list = column_list
        self.total_record = total_record
        self.freq_df = None
        
    def process(self)->pd.core.frame.DataFrame:
        self.make_clean_freq_table()
        self.add_ratio()
        if self.total_record:
            self.add_total_record()
        return self.freq_df
        
    def make_clean_freq_table(self):
        freq_sr = self.df[self.column_list].value_counts()
        freq_sr.sort_index(inplace=True)
        freq_df = freq_sr.reset_index(drop=False)
        freq_df.set_index(self.column_list, inplace=True)
        self.freq_df = freq_df
        
    def add_ratio(self):
        total = self.freq_df['count'].sum()
        self.freq_df['ratio'] = (self.freq_df['count']/total).round(2)
        
    def add_total_record(self):
        sum_record = pd.DataFrame(self.freq_df.sum()).T
        # add index
        index_size = len(self.freq_df.index[0])
        multi_index = tuple(["total" for i in range(index_size)])
        sum_record.index = pd.MultiIndex.from_tuples([multi_index])
        # add total record
        self.freq_df = pd.concat([self.freq_df, sum_record])
        # count dtype exchange to integer
        self.freq_df['count'] = self.freq_df['count'].astype(np.int64)
#####################################################################################





# 시각화에 자주 쓰이는 Code
#####################################################################################
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_RGB_image_by_cv2(image_path:str, RGB=False)->np.ndarray:
    img_arr = cv2.imread(image_path, cv2.IMREAD_COLOR)   # COLOR image로 불러온다.
    if RGB:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    return img_arr



def draw_img_and_bbox_torch_style(img, bbox_list, _figsize=(12, 8), _edgecolor="red", none_axis=True, title=None):

    def convert_xywh(bbox):
        
        x1, y1, x2, y2 = bbox
        if x1 <= x2:
            scaled_bbox = [x1, y1, x2, y2]
        else:
            scaled_bbox = [x2, y2, x1, y1]
        x = scaled_bbox[0]
        y = scaled_bbox[1]
        width = scaled_bbox[2] - scaled_bbox[0]
        height = scaled_bbox[3] - scaled_bbox[1]

        return x, y, width, height
    
    fig, ax = plt.subplots(figsize=_figsize)
    ax.imshow(img)
    for bbox in bbox_list:
        x, y, width, height = convert_xywh(bbox)
        ax.add_patch(patches.Rectangle((x, y), width, height,edgecolor="red", fill =False))
    if none_axis:
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    if title is not None:
        plt.title(title, fontsize = 20, pad = 20, color="limegreen")
        
    plt.show()
#####################################################################################