import random
import numpy as np

from Module.augmentation.random_txt_and_bbox_maker import add_random_string
from Module.augmentation.reduce_data_size import reduce_image_and_bbox
from Module.augmentation.flip import flip_img_and_bbox



class my_augmentation:
    
    def __init__(
        self,
        add_txt_p=0.0, reduce_size_p=0.0, flip_p=0.0,
        rs_param_dict={
            "num_range":range(1,5),
            "size_range":np.arange(0.2, 1.0, 0.05),
            "str_patn_list":[
                r"[A-Za-fh-ik-or-xz0-9]{1,2}",
                r"[A-Za-fh-ik-or-xz0-9]{1,3}[A-Za-fh-ik-or-xz0-9 _-]{0,3}[A-Za-fh-ik-or-xz0-9]{0,3}"
            ]
        },
        reduce_max=2.0,
        flip_key_list=["horizontal", "vertical", "all"]
    ):
        self.add_txt_p = add_txt_p
        self.reduce_size_p = reduce_size_p
        self.flip_p = flip_p
        self.rs_param_dict = rs_param_dict
        self.reduce_max = reduce_max
        self.flip_key_list = flip_key_list
        
        
    def process(self, image, target):
        
        if random.random() < self.add_txt_p:
            image, target = add_random_string(
                image, target,
                num_range=self.rs_param_dict["num_range"],
                size_range=self.rs_param_dict["size_range"],
                str_patn_list=self.rs_param_dict["str_patn_list"]
            ).process()
            
        if random.random() < self.reduce_size_p:
            reduce_size = random.uniform(1, self.reduce_max)
            image, target = reduce_image_and_bbox(image, target, ratio=reduce_size).process()
            
        if random.random() < self.flip_p:
            flip_key = random.choice(["horizontal", "vertical", "all"])
            image, target = flip_img_and_bbox(image, target).process(key=flip_key)
        
        return image, target