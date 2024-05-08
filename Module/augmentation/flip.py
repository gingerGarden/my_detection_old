import cv2
from copy import deepcopy



class flip_img_and_bbox:
    
    def __init__(self, image, target):
        
        self.image = image
        self.target = deepcopy(target)
        
        
    def process(self, key="all"):
        # 좌우반전
        if key == "horizontal":
            self.horizontal_flip()
        # 상하반전
        elif key == "vertical":
            self.vertical_flip()
        # 상하좌우반전
        else:
            self.horizontal_flip()
            self.vertical_flip()
        return self.image, self.target
        
        
    # 좌우 반전
    def horizontal_flip(self):
        # 이미지의 좌우 반전
        self.image = cv2.flip(self.image, 1)
        # boxes의 좌우 반전
        _, width, _ = self.image.shape
        stack_list = []
        for bbox in self.target["boxes"]:

            x1 = width - bbox[0]
            x2 = width - bbox[2]
            new_bbox = [x2, bbox[1], x1, bbox[3]]
            stack_list.append(new_bbox)

        self.target["boxes"] = stack_list
        
        
    # 상하 반전
    def vertical_flip(self):
        self.image = cv2.flip(self.image, 0)
        # boxex의 상하 반전
        height, _, _ = self.image.shape
        stack_list = []
        for bbox in self.target["boxes"]:

            y1 = height - bbox[1]
            y2 = height - bbox[3]
            new_bbox = [bbox[0], y2, bbox[2], y1]
            stack_list.append(new_bbox)

        self.target["boxes"] = stack_list