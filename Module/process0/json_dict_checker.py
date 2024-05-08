import pandas as pd
from Module.Global_variable import BASIC_KEY_DF




class json_dict_checker:
    
    def __init__(self, json_dict, basic_key_df=BASIC_KEY_DF):
        
        self.json_dict = json_dict
        self.raw_df = pd.DataFrame(json_dict['images'])
        self.basic_key_df = basic_key_df
        self.have_bbox_df = self.raw_df[self.raw_df["bbox"] == "True"]
        self.bbox_df = None

        
    def make_bbox_df(self):

        stack_df = pd.DataFrame()
        for idx in self.have_bbox_df.index:

            file_name = self.raw_df.loc[idx, "file_name"]
            height = self.raw_df.loc[idx, "height"]
            width = self.raw_df.loc[idx, "width"]
            
            bbox_df = pd.DataFrame(self.json_dict['annotations'][file_name]['bbox'])
            bbox_df.columns = ['x_min', 'y_min', 'x_max', 'y_max']
            bbox_df["file_name"] = file_name
            bbox_df["height"] = height
            bbox_df["width"] = width

            stack_df = pd.concat([stack_df, bbox_df])

        # 원본 파일명을 붙인다.
        self.bbox_df = self.add_original_filename_bbox_df(stack_df.reset_index(drop=True))

        
    def add_original_filename_bbox_df(self, bbox_df):

        bbox_df["merging_key"] = bbox_df["file_name"].str.split(".", expand=True)[0]
        merge_df = pd.merge(bbox_df, self.basic_key_df[["new_filekey", "old_filename"]], left_on="merging_key", right_on="new_filekey")
        del(merge_df["merging_key"])
        del(merge_df["new_filekey"])

        return merge_df

    
    def check1_bboxs_is_out_of_images(self):

        print("1. json_dict에 있는 모든 bbox의 최대 좌표가 해당 이미지의 바깥에 있는지 확인.")
        print("----"*20)
        self.check_bbox_is_out_of_image(check="x_max")
        self.check_bbox_is_out_of_image(check="y_max")
        print("\n")
    
    
    def check_bbox_is_out_of_image(self, check="x_max"):

        compare = "width" if check == "x_max" else "height"
        strange = self.bbox_df[self.bbox_df[check] > self.bbox_df[compare]]

        if len(strange) > 0:
            show_markdown_df(strange)
        else:
            print(f"bboxs의 {check}는 각 image의 {compare} 보다 작다")
            

    def check2_maximum_size_of_bboxs_is_smaller_than_size_of_images(self):
        
        print("2. json_dict에 있는 모든 bbox 중 이미지를 완전히 둘러싸고 있는 경우가 있는지 확인.")
        print("----"*20)
        # bbox가 image보다 큰 경우가 있는지
        bbox_more_bigger_than_image = self.bbox_df[
            (self.bbox_df["x_max"] - self.bbox_df["x_min"]) * (self.bbox_df["y_max"] - self.bbox_df["y_min"])\
            > (self.bbox_df["height"] * self.bbox_df["width"])
        ]
        if len(bbox_more_bigger_than_image) > 0:
            show_markdown_df(bbox_more_bigger_than_image)
        else:
            print("모든 bbox의 넓이는 이미지의 넓이보다 작다")
        print("\n")
        
        
    def check3_bboxs_all_x_min_smaller_than_x_max(self):

        print("3. json_dict에 있는 모든 bbox의 x_min이 x_max보다 작은지 확인")
        print("----"*20)
        x_min_bigger_than_x_max = self.bbox_df[self.bbox_df["x_min"] > self.bbox_df["x_max"]]
        target_size = len(x_min_bigger_than_x_max)
        if target_size != 0:
            print("해당 bbox들은 x_min이 x_max보다 크다.")
            show_markdown_df(x_min_bigger_than_x_max)
        else:
            print("모든 bbox들은 x_min이 x_max보다 작다.")
        print("\n")
            
        
    def check4_bboxs_all_y_min_smaller_than_y_max(self):

        print("4. json_dict에 있는 모든 bbox의 y_min이 y_max보다 작은지 확인")
        print("----"*20)
        y_min_bigger_than_y_max = self.bbox_df[self.bbox_df["y_min"] > self.bbox_df["y_max"]]
        target_size = len(y_min_bigger_than_y_max)
        if target_size != 0:
            print("해당 bbox들은 y_min이 y_max보다 크다.")
            show_markdown_df(y_min_bigger_than_y_max)
        else:
            print("모든 bbox들은 y_min이 y_max보다 작다.")