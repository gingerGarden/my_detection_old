import cv2
import numpy as np
from copy import deepcopy



class reduce_image_and_bbox:
    
    def __init__(self, image, target, ratio):
        
        self.image = image
        self.target = deepcopy(target)
        self.ratio = ratio
        
        
    def process(self):
        # image의 크기를 줄인다.
        self.reduce_image_size()
        # bbox의 크기를 줄인다.
        self.reduce_bbox()
        # 축소된 bbox의 크기에 맞게 area의 값을 변경한다.
        self.area_update()
        return self.image, self.target
        
    

    def reduce_image_size(self):

        height, width, _ = self.image.shape
        # 내림하여 정수처리
        new_height = int(np.trunc(height/self.ratio))
        new_width = int(np.trunc(width/self.ratio))
        self.image = cv2.resize(self.image, (new_width, new_height), cv2.INTER_LANCZOS4)
    
    
    def reduce_bbox(self):
        
        stack_list = []
        for bbox in self.target["boxes"]:
            reduce_box = np.array(bbox)/self.ratio
            reduce_box = list(np.trunc(reduce_box).astype(np.int64))
            stack_list.append(reduce_box)
        self.target["boxes"] = stack_list
    
    
    def area_update(self):
        
        stack_list = []
        for bbox in self.target["boxes"]:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            stack_list.append(area)

        self.target["area"] = stack_list
        