import pandas as pd
import numpy as np

from Module.Global_variable import IMG_PARENTS_PATH, BASIC_KEY_DF, DATASET_STRING
from Module.utils.Convenience_Function import get_RGB_image_by_cv2, draw_img_and_bbox_torch_style



class model_predict_checker:
    
    def __init__(self, predict_dict, k, idx_dict, img_root=IMG_PARENTS_PATH, key_df=BASIC_KEY_DF, dataset_str=DATASET_STRING):
        
        self.pred_dict = predict_dict
        self.k = k
        self.idx_dict = idx_dict
        self.dataset_str = dataset_str
        self.img_root=img_root
        self.key_df = key_df
        # 1차 생성
        self.label_dict = None
        self.pred_id_box_df = None
        self.label_id_box_df = None
        # 2차 생성
        self.none_GT_over_pred_df = None    # 실제보다 과하게 예측한 경우.
        self.GT_not_found_id_df = None      # 모델이 찾았어야 하지만 찾지 못한 경우.
        self.diff_id_df = None              # 모델이 찾은 bbox와 찾지 못한 bbox의 갯수가 다른 경우.
        self.good_id_df = None              # 모델이 찾은 bbox와 찾지 못한 bbox의 갯수가 동일한 경우
        
        
        
    def make_basic_value(self):

        # label_dict을 생성한다.
        self.make_label_dict()
        # pred_id_box_df를 생성한다 - 예측한 img_id와 box의 수
        self.make_pred_id_box_df()
        # label_id_box_df를 생성한다. 실제 box를 가지고 있는 img_id와 box의 수
        self.make_label_id_box_df()
        # 모델이 과하게 찾아낸 case를 정리한다.
        self.find_over_predict_id_df()
        # 모델이 찾지 못한 case를 정리한다.
        self.find_GT_not_found_id_df()
        # 모델이 찾은 bbox와 찾지 못한 bbox의 갯수가 다른 경우를 정리한다.
        self.make_diff_id_df()
        # 결과 요약 및 DataFrame을 내보낸다.
        self.print_predict_descript()
        
        over_pred_df = self.none_GT_over_pred_df
        under_pred_df = self.GT_not_found_id_df
        diff_pred_df = self.diff_id_df
        good_id_df = self.good_id_df
        
        return over_pred_df, under_pred_df, diff_pred_df, good_id_df
        
        
        
    # self.label_dict 생성
    def make_label_dict(self):
        
        test_label_dict = self.idx_dict[f"{self.dataset_str}{self.k}"]['test']
        test_imgs_df = pd.DataFrame(test_label_dict['images'])

        self.label_dict = {
            'images':test_imgs_df.set_index('id').to_dict('index'),
            'annotations':test_label_dict['annotations']
        }
        
        
        
    # 모델이 예측한 것 중 bbox가 있는 것들과 box의 갯수를 가지고 온다.
    def make_pred_id_box_df(self):

        stack_list = []
        for img_id, item in self.pred_dict.items():
            boxes_size = len(item['boxes'])
            if boxes_size > 0:
                record = (img_id, boxes_size)
                stack_list.append(record)

        self.pred_id_box_df = pd.DataFrame(stack_list, columns=['img_id', 'pred_size'])
        
        
        
    def make_label_id_box_df(self):
        
        stack_list = []
        for img_id, item in self.label_dict['images'].items():

            anno_key = item['file_name']
            box_size = len(self.label_dict['annotations'][anno_key]['bbox'])

            if box_size > 0:
                record = (img_id, box_size)
                stack_list.append(record)

        self.label_id_box_df = pd.DataFrame(stack_list, columns=['img_id', 'label_size'])
        
        
        
    # 모델이 과하게 예측한 Case
    def find_over_predict_id_df(self):

        over_pred_id_set = list(
            set(self.pred_id_box_df["img_id"]) - set(self.label_id_box_df["img_id"])
        )
        self.none_GT_over_pred_df = self.pred_id_box_df[
            self.pred_id_box_df["img_id"].isin(over_pred_id_set)
        ]
        
        
        
    # 모델이 예측했어야 했으니 찾지 못한 Case
    def find_GT_not_found_id_df(self):

        target_id_list = list(
            set(self.label_id_box_df["img_id"]) - set(self.pred_id_box_df["img_id"])
        )
        self.GT_not_found_id_df = self.label_id_box_df[
            self.label_id_box_df["img_id"].isin(target_id_list)
        ]
        
        
        
    # 모델이 찾은 갯수와 모델이 못찾은 갯수가 다른 경우
    def make_diff_id_df(self):

        merge_id_df = pd.merge(self.label_id_box_df, self.pred_id_box_df, on='img_id')
        merge_id_df["deviation"] = merge_id_df["label_size"] - merge_id_df["pred_size"]

        self.diff_id_df = merge_id_df[merge_id_df['deviation'] != 0]
        self.good_id_df = merge_id_df[merge_id_df['deviation'] == 0]
        
        
        
    def print_predict_descript(self):

        total_images = len(self.label_dict['images'])
        bbox_have_images = len(self.label_id_box_df)
        print(f"1. 추론에 사용된 전체 이미지의 갯수: {total_images}")
        print("----"*20)
        print(f"1-1. GT를 가지고 있는 이미지의 갯수: {bbox_have_images}({bbox_have_images/total_images:.3f})")
        none_bbox_imgs = total_images - bbox_have_images
        print(f"1-2. GT를 가지고 있지 않은 이미지의 갯수: {none_bbox_imgs}({none_bbox_imgs/total_images:.3f})")
        print('\n')
        print("2. 모델의 추론 능력")
        print("----"*20)
        print(f"2-1. 모델이 GT를 찾지 못한 경우: {len(self.GT_not_found_id_df)}")
        print(f"2-2. 모델이 GT가 없는 대상을 과하게 찾은 경우: {len(self.none_GT_over_pred_df)}")
        total_GT_size = self.label_id_box_df['label_size'].sum()
        print(f"2-3. 총 GT의 크기 = {total_GT_size}")
        coudnt_find_GT = self.diff_id_df[self.diff_id_df['deviation'] > 0]['deviation'].sum()
        print(f"2-3. 모델이 찾지 못한 GT의 크기 = {coudnt_find_GT}({coudnt_find_GT/total_GT_size:.3f})")
        count_wrong_GT = np.abs(self.diff_id_df[self.diff_id_df['deviation'] < 0]['deviation'].sum())
        print(f"2-4. 모델이 과도하게 찾은 GT의 크기 = {count_wrong_GT}")

    
        
    def draw_predict_image(self, img_id):
        
        img_name = self.label_dict["images"][img_id]['file_name']
        img_path = f"{self.img_root}/{img_name}"
        img = get_RGB_image_by_cv2(img_path, RGB=True)
        pred_boxes = self.pred_dict[img_id]['boxes']

        # 원본 파일명을 가지고 온다.
        origin_file_name = self.key_df[
            self.key_df["new_filekey"] == img_name.split(".")[0]
        ]['old_filename'].item()

        title = f"{origin_file_name}"
        
        # 이미지 표현
        draw_img_and_bbox_torch_style(img, bbox_list=pred_boxes, title=title)