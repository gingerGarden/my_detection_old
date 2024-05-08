import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Module.utils.Convenience_Function import get_RGB_image_by_cv2


class draw_bbox_my_torch_style:
    
    
    def __init__(self, json_dict, have_bbox_df, img_parents_path, fig_size=(12, 8), edgecolor="red", titlecolor="limegreen", none_axis=True):
        
        self.img_dict = json_dict['images']
        self.anno_dict = json_dict['annotations']
        self.img_parents_path = img_parents_path
        self.have_bbox_df = have_bbox_df
        self.fig_size = fig_size
        self.edgecolor = edgecolor
        self.titlecolor = titlecolor
        self.none_axis = none_axis


    def draw_idx_image(self, idx):

        # image의 경로와 bbox를 가지고 온다.
        img_path, bbox_list = self.get_target_idx_img_path_and_bbox_list(idx)
        img = get_RGB_image_by_cv2(image_path=img_path, RGB=True)

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.imshow(img)

        for bbox in bbox_list:

            x, y, width, height = self.convert_xywh(bbox)
            ax.add_patch(patches.Rectangle((x, y), width, height,edgecolor="red", fill =False))

        plt.title(f"image shape: {img.shape}", fontsize = 20, pad = 20, color=self.titlecolor)

        if self.none_axis:
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)

        plt.show()
        
        
    def draw_random_images(self, sample_size):

        random_index = np.random.choice(self.have_bbox_df.index, size=sample_size, replace=False)

        for idx in random_index:

            self.draw_idx_image(idx)
        
        
    def get_target_idx_img_path_and_bbox_list(self, idx):

        # img_dict을 가지고 온다.
        img_dict = self.img_dict[idx]
        img_name = img_dict['file_name']
        img_path = f"{self.img_parents_path}/{img_name}"

        # bbox 존재 확인
        if bool(img_dict['bbox']):
            anno_dict = self.anno_dict[img_name]
            bbox_list = anno_dict['bbox']
        else:
            bbox_list = None

        return img_path, bbox_list
    
    
#     def convert_xywh(self, bbox):
        
#         x = bbox[0]
#         y = bbox[1]
#         width = bbox[2] - bbox[0]
#         height = bbox[3] - bbox[1]

#         return x, y, width, height
    
    
    def convert_xywh(self, bbox):
        
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        
        if x1 <= x2:
            scaled_bbox = [x1, y1, x2, y2]
        else:
            scaled_bbox = [x2, y2, x1, y1]
            
        x = scaled_bbox[0]
        y = scaled_bbox[1]
        width = scaled_bbox[2] - scaled_bbox[0]
        height = scaled_bbox[3] - scaled_bbox[1]

        return x, y, width, height