import rstr
from copy import deepcopy
import numpy as np
import cv2



class add_random_string:
    """
    txt size가 0.3 이하일 때, thickness는 무조건 1이다.
    """
    def __init__(
        self, image, target,
        num_range=range(1, 5),
        size_range=np.arange(0.2, 1.0, 0.05),
        str_patn_list=[
            r"[A-Za-fh-ik-or-xz0-9]{1,2}",
            r"[A-Za-fh-ik-or-xz0-9]{1,3}[A-Za-fh-ik-or-xz0-9 _-]{0,3}[A-Za-fh-ik-or-xz0-9]{0,3}"
        ],   # 'g','j','p','q','y'는 bbox의 아래를 뚫고 갈 수 있으므로 일괄 제외
        bbox_margin=4,
        labels=1, iscrowd=0,
    ):  
        self.image = image
        self.target = deepcopy(target)
        self.labels = labels
        self.iscrowd = iscrowd
        
        self.num_range = num_range
        self.size_range = size_range
        self.str_patn_list = str_patn_list
        self.bbox_margin = bbox_margin
        
        self.color_dict = {
            "blue":(255, 0, 0),
            "green":(0, 255, 0),
            "red":(0, 0, 255),
            "white":(255, 255, 255),
            "black":(0, 0, 0)
        }
        self.font_list = [
            cv2.FONT_HERSHEY_SIMPLEX,         # 0
#             cv2.FONT_HERSHEY_PLAIN,           # 1
            cv2.FONT_HERSHEY_DUPLEX,          # 2
            cv2.FONT_HERSHEY_COMPLEX,         # 3
            cv2.FONT_HERSHEY_TRIPLEX,         # 4
#             cv2.FONT_HERSHEY_COMPLEX_SMALL,   # 5
#             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,  # 6
#             cv2.FONT_HERSHEY_SCRIPT_COMPLEX,  # 7
            cv2.FONT_ITALIC                   # 16
        ]
        
        
    def process(self):

        # 몇 개의 txt를 생성할 것인가
        iter_range = np.random.choice(self.num_range, size=1).item()

        for i in range(iter_range):

            img_key = np.random.choice(["white", "black", "black"], size=1).item()

            # y,p,q는 글씨가 bbox 아래로 나옴. fontsize에 비래해서 y2에 추가 margin을 주자.
            if img_key == "black":
                x, y, txt_size = self.add_txt_to_image(key="black")
            else:
                x, y, txt_size = self.add_txt_to_image()

            # target update
            self.target_update(x, y, txt_size)

        return self.image, self.target
        
        
    
    # image내 text 생성 알고리즘
    ################################################################################
    # 문자 안에 이미지 삽입
    def add_txt_to_image(self, key="white"):
        # 무작위 txt 생성
        txt = self.get_random_txt()
        # font와 fontsize, thickness를 정의한다.
        font, _fontScale, _thichness = self.get_font_and_font_scale()
        # font와 txt의 크기를 가지고 온다.
        if key == "black":
            txt_size = self.get_font_and_text_size(txt, font, _fontScale, _thichness+1)
        else:
            txt_size = self.get_font_and_text_size(txt, font, _fontScale, _thichness)
        # 무작위 text 생성 위치를 가지고 온다.
        x, y, y_under = self.get_txt_start_point(txt_size)
        # 이미지에 txt를 삽입.
        ## cv2에서 putText는 좌측 하단을 기준으로 글씨 삽입
        if key == "black":
            self.image = cv2.putText(self.image, txt, (x, y_under), font, _fontScale, self.color_dict["black"], _thichness+1)
        self.image = cv2.putText(self.image, txt, (x, y_under), font, _fontScale, self.color_dict["white"], _thichness)
        
        return x, y, txt_size
    
    
    # 무작위 text 생성
    def get_random_txt(self):
        txt_pattern = np.random.choice(self.str_patn_list, size=1).item()   # text의 길이
        return rstr.xeger(txt_pattern)
    
#     # 무작위 text 생성
#     def get_random_txt(self):

#         txt_len = np.random.choice(self.len_range, size=1).item()   # text의 길이
#         txt_pattern = self.str_patn + "{" + f"{txt_len}" + "}"
#         return rstr.xeger(txt_pattern)
    
    
    
    def get_font_and_font_scale(self):
        font = np.random.choice(self.font_list, size=1).item()
        _fontScale = np.random.choice(self.size_range, size=1).item()
        if _fontScale >= 0.5:
            _thichness = np.random.choice([1,2], size=1).item()
        else:
            _thichness = 1
        return font, _fontScale, _thichness
    
    
    def get_font_and_text_size(self, txt, font, _fontScale, _thichness):
        text_size = cv2.getTextSize(
            txt, font, fontScale=_fontScale, thickness=_thichness
        )[0]
        return text_size
    
    
    def get_txt_start_point(self, txt_size):

        # txt의 무작위 시작 좌표
        height, width, _ = self.image.shape

        # txt의 크기를 고려해 margin px 추가
        max_width = (width - txt_size[0]) - self.bbox_margin
        max_height = (height - txt_size[1]) - self.bbox_margin

        # txt의 무작위 좌표 계산 - x, y는 좌측 하단이므로 조정 필요
        x = np.random.choice(range(self.bbox_margin, max_width), size=1).item()
        y = np.random.choice(range(self.bbox_margin, max_height), size=1).item()
        y_under = y + txt_size[1] 

        return x, y, y_under
    ################################################################################
    
    
    # target 업데이트 알고리즘
    ################################################################################
    def target_update(self, x, y, txt_size):

        # 신규 bbox 계산
        new_bbox = self.make_new_bbox(x, y, txt_size)
        # bbox의 넓이
        new_bbox_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])

        self.target['boxes'].append(new_bbox)
        self.target['labels'].append(self.labels)
        self.target['area'].append(new_bbox_area)
        self.target['iscrowd'].append(self.iscrowd)
    
    
    # bbox 추가
    def make_new_bbox(self, x, y, txt_size):
        """
        self.bbox_margin보다 margion만큼 덜 뺀다. 이는 1로 한다.
        """
        bbox_margin = self.bbox_margin - 1
        new_bbox = [
            x-bbox_margin,
            y-bbox_margin,
            x+txt_size[0]+(bbox_margin),
            y+txt_size[1]+(bbox_margin+1)
        ]
        return new_bbox
    ################################################################################
