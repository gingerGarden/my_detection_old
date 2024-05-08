import os, time, datetime, json, warnings
import numpy as np
from sys import stdout
from collections import deque




# 실수로 표현된 초를 문자열 시간으로 변환
def decimal_seconds_to_time_string(decimal_s):
    """
    실수로 표현된 초(decimal seconds)를 이해하기 쉬운 문자형으로 변환한다.
    """
    time_delta = datetime.timedelta(seconds=decimal_s)
    str_time_delta = str(time_delta).split(".")
    time1 = str_time_delta[0]
    if len(str_time_delta) == 1:
        time2 = "00"
    else:
        time2 = str_time_delta[1][:2]
    return f"{time1}.{time2}"




# 하나의 list로 구성된 json log 파일을 다룬다.
#################################################################################################
# []로 구성된 json log 파일 생성
def make_new_json_list_log_file(log_path):
    """
    list형 하나만 들어가 있는 기초 json 파일을 생성한다.
    """
    with open(log_path, 'w') as file:
        json.dump([], file)
            
            
            
# log 파일을 불러와서 신규 로그를 추가한다.
def append_log_dict_to_json_list_log_file(log_path, log_dict):
    """
    `make_new_list_json_log_file(log_path)`를 통해 생성된 list만 들어있는 기본 json file을
    불러오고, 그 안에 log_dict을 추가한다.
    """
    if type(log_dict) == dict:
        # json file을 읽는다.
        with open(log_path, 'r') as file:
            data = json.load(file)
        # 신규 log 추가
        data.append(log_dict)
        # 변경된 내용 저장
        with open(log_path, 'w') as file:
            json.dump(data, file)  # indent=4는 들어쓰기로, json의 가독성을 올려준다.
    else:
        warnings.warn(
            "TypeError: The function only accepts logs created as a dictionary. Please check the type of the log."
        )
#################################################################################################
        



# Log progress bar
#################################################################################################
class my_time_log_progressbar:
    """
    progressbar를 생성한다.
    =======================================================================
    * `basic(it)`: 기초적인 progressbar 생성
    * `with_load_delta(it)`: iteration에서 데이터를 꺼내오는 시간을 함께 출력
    """
    def __init__(self, header="", bar_size=60, verbose=True, sep_next=True, num_size=5):
        
        self.header = header
        self.bar_size = bar_size
        self.verbose = verbose
        self.sep_next = sep_next      # progressbar를 다음 문장과 분할할 것인지
        self.num_size = num_size      # progressbar에 진행되는 수치 ( n/ N)의 맞춤형 길이
        self.iter_size = None         # iteration의 길이
        self.now_iter = None          # 현재 iteration의 번호
        self.load_ins = None          # 데이터를 가지고 오는데 걸리는 시간
        self.eta_ins = None           # 해당 iter가 끝날 때까지 걸릴 것으로 예상되는 시간
        self.one_iter_ins = None      # 데이터를 가지고 오는 시간부터 iter가 끝나는 시간까지
        self.stack_time = 0           # process 완료까지 누적 시간
        
        
        
    def basic(self, it):
        self.iter_size = len(it)
        if self.verbose: self.print_progress_bar(now_iter=0)
            
        for self.now_iter, item in enumerate(it):
            yield item
            if self.verbose: self.print_progress_bar(now_iter=self.now_iter+1)
                
        if self.verbose: self.print_txt("\n")
        
        
        
    def with_time_log(self, it):
        self.iter_size = len(it)
        if self.verbose: self.print_progress_bar(now_iter=0)
            
        # iterator에서 item을 가져오는 시간 계산
        self.load_ins = calculate_time_log(deque_key="load")
        self.load_ins.update(time.time())
        
        # iteration에서 한 process 종료까지 걸리는 시간
        self.one_iter_ins = calculate_time_log(deque_key="process")
        self.one_iter_ins.update(time.time())
        
        # iteration가 포함된 전체 프로세스 종료까지 남은 예상 시간
        self.eta_ins = calculate_time_log(deque_key="eta")
        
        for self.now_iter, item in enumerate(it):
            # iteration에서 data를 load하는데 걸린 시간 측정
            self.load_ins.update(time.time())
            yield item
            if self.verbose: self.print_progress_bar(now_iter=self.now_iter+1)
            
        # verbose나 sep_next 둘 다 True인 경우에만 내린다.
        if self.verbose and self.sep_next:
            self.print_txt("\n")  
        
        
        
    # iterator가 들어가는(예, for문)의 끝에 넣는다.
    def put_it_at_the_end_of_the_iteration_process(self):

        # 이번 iteration에서 data를 load하는데 걸린 시간.
        load_time = self.load_ins.time_delta()
        
        # 한 iteration 안에서 process가 완료되는 시간(data load 시간 제외)
        self.load_ins.update(time.time())               # 한 iteration의 process가 완료 됐으므로, update한다.
        iter_process_time = self.load_ins.time_delta()  
        
        # data를 load하는 것을 포함한 process가 완료된 후 update된 시간과의 차이
        self.one_iter_ins.update(time.time())
        process_time = self.one_iter_ins.time_delta()
        
        # iteration process 예상 종료 시간
        self.eta_ins.update(float(process_time))
        eta = self.estimated_time_of_arrival()
        
        # process 종료까지 누적 시간
        self.stack_time += float(process_time)
        clean_time = decimal_seconds_to_time_string(decimal_s=self.stack_time)
        
        result = {
            "eta":eta,
            "stack":clean_time,
            "process":process_time,
            "data_load":load_time,
            "iter_train":iter_process_time
        }
        return result
        
        
        
    def estimated_time_of_arrival(self):
        # 한 iteration당 소모된 간격 시간
        aver_iter_delta = np.mean(self.eta_ins.time_deque)
        # 남아있는 iteration의 길이
        rest_itration_size = self.iter_size - (self.now_iter + 1)
        # 예상 예측 시간
        predict_rest_time = aver_iter_delta * rest_itration_size
        # 깔끔한 문자로 정리
        result = decimal_seconds_to_time_string(decimal_s=predict_rest_time)
        return result
        
        
        
    def print_progress_bar(self, now_iter):
        done_size = int(self.bar_size*(now_iter)/self.iter_size)
        rest_size = self.bar_size - done_size
        
        count = f"({now_iter:>{self.num_size}}/{self.iter_size:>{self.num_size}})"
        
        progress_txt = "\r%s[%s%s] " % (
            self.header, (done_size+1)*"=", rest_size*".")
        progress_txt = progress_txt + count
        self.print_txt(progress_txt)
        
        
        
    def print_txt(self, txt):
        stdout.write(txt)
        stdout.flush()
        
        
          
# 시간 관련 log 계산
class calculate_time_log:
    
    def __init__(self, deque_key="load", delta_format="{:.4f}"):
        """
        dtype `collections.deque` 를 이용한 시간 관련 로그 계산 class
        -------------------------------------------------------------------------------------
        deque_key="load":
        >>> load로 설정 시, deque의 maxlen은 2로 설정
        >>> eta로 설정 시, deque의 maxlen은 10으로 설정
        
        `defined_deque_maxlen(deque_key)`: `deque`의 최대 길이 정의
        `update(value)`: `deque` 오른쪽에 새로운 값 입력, maxlen을 넘는 경우 왼쪽 하나 제거
        `time_delta()`: `deque`의 2개 인자(before, after)의 차이
        
        # `collections.deque`란?
        deque(double-ended queue)는 collections 모듈에서 제공하는 컨테이너 데이터 타입의 하나로,
        양 끝에서 요소를 빠르게 추가하거나 제거할 수 있도록 설계된 일종의 큐이다. deque는 내부적으로
        Linked list 구조를 사용하여 구현되어 있으므로, 양 끝에서 연산이 매우 효율적이다.
        --------------------------------------------------------------------------------------
        1. 양방향 연산
           - `deque`는 양방향 연산이 쉬움. 양 끝에 element를 추가하거나 제거 쉽게 가능.
           - 주요 method 
             > `append()`: 오른쪽 끝에 element 추가
             > `appendleft()`: 왼쪽 끝에 element 추가
             > `pop()`: 오른쪽 끝 element 제거
             > `popleft()`: 왼쪽 끝 element 제거
       2. 고정 길이 설정 가능.
          - 새 element 추가 시, `deque`의 최대 길이를 초과하면 반대편 끝의 element가 자동으로 제거
            (고정된 크기의 버퍼 유지 시 유용).
       3. 인덱싱과 반복
          - `deque`는 인덱싱과 반복이 가능하나, list에 비해 인덱싱 속도가 느림.
          - 반복 작업에는 적합하나, 빈번한 임의 접근이 필요한 경우 일반 list가 더 적합할 수 있음.
       4. 스레드 안전(Thread-safe)
          - `deque`는 멀티스레딩 환경에서 사용하기 적합하며, 여러 스레드가 동시에 `deque`에 접근해
            수정하는 것이 가능함.
        """
        self.time_deque = deque(maxlen=self.defined_deque_maxlen(deque_key))
        self.delta_format = delta_format
        
        
        
    def defined_deque_maxlen(self, deque_key):
        """
        load, process는 2개 eta는 10개
        """
        return 2 if deque_key in ["load", "process"] else 20

    
    
    # deque의 오른쪽에 value 입력, 만약 deque가 가득 찬 경우 왼쪽부터 하나씩 제거 됨.
    def update(self, value):
        self.time_deque.append(value)

        
        
    def time_delta(self):
        delta = self.time_deque[1] - self.time_deque[0]
        result = self.delta_format.format(delta)
        return result
#################################################################################################