from Module.utils.Convenience_Function import read_json, get_RGB_image_by_cv2
from Module.Global_variable import pd, np, tqdm



# image dictionary 생성
###############################################################################
class make_image_dictionary:
    
    def __init__(self, df, img_parents_path):
        
        self.df = df
        self.img_parents_path = img_parents_path


    def process(self):

        stack_list = []
        for idx in tqdm(self.df.index):
            target_record = self.df.loc[idx]
            stack_list.append(self.get_image_dict(idx))
        return stack_list

    
    def get_image_dict(self, idx):

        imgName = self.df.loc[idx, "new_filekey"]
        imgName = f"{imgName}.jpg"
        imgPath = f"{self.img_parents_path}/{imgName}"
        img = get_RGB_image_by_cv2(imgPath)
        result = {
            "file_name":imgName,
            "height":img.shape[0],
            "width":img.shape[1],
            "id":idx,
            "bbox":str(self.df.loc[idx, "txt_json"])
        }
        return result
###############################################################################



# Annoation dictionary 생성
###############################################################################
class make_annotation_dict:
    
    
    def __init__(self, key_df, json_path):
        
        self.key_df = key_df
        self.json_path = json_path
    
    
    def process(self):

        annoation_dict = dict()
        for idx in self.key_df.index:

            annoation_key = self.key_df.loc[idx, "new_filekey"] + ".jpg"
            annoation_dict[annoation_key] = self.make_annotation_element(idx)

        return annoation_dict


    def make_annotation_element(self, idx):

        txt_have = self.key_df.loc[idx, "txt_json"]
        if txt_have:
            annotation_value = self.make_annotation_element_if_label_exists(idx)
        else:
            annotation_value = {
                'bbox':[],
                'area':[],
                'category_id':[],
                'iscrowd':[]
            }
        return annotation_value

    
    def make_annotation_element_if_label_exists(self, idx):

        filekey = self.key_df.loc[idx, "new_filekey"]
        jsonPath = f"{self.json_path}/{filekey}.json"

        # bbox를 가지고 오고 정리한다. - x1, y1은 내림, y1, y2은 올림하여 정수로 만든다. - object가 잘리지 않게 최대 1px이하로 영역 늘림.
        bbox_list = self.make_scaled_bbox_list(read_json(jsonPath))
        # area list 생성
        area_list = self.make_area_list(bbox_list)
        # category - 모두 txt임.
        cate_list = [1 for i in range(len(area_list))]
        # iscrowd - 모두 군중 상태가 아닌 것을 가정함.
        iscrowd_list = [0 for i in range(len(area_list))]
        result = {
            "bbox":bbox_list,
            "area":area_list,
            "category_id":cate_list,
            "iscrowd":iscrowd_list,
        }
        return result
    

#     def make_scaled_bbox_list(self, bbox_list_dict):

#         stack_list = []
#         for bbox_key in bbox_list_dict.keys():

#             bbox_list = bbox_list_dict[bbox_key]
#             x1 = np.trunc(bbox_list[0][0])
#             y1 = np.trunc(bbox_list[0][1])
#             x2 = np.trunc(bbox_list[1][0])
#             y2 = np.trunc(bbox_list[1][1])
            
#             # x1과 x2중 작은 것을 쉬작 위치로 함.
#             # [x_min, y_min, x_max, y_max]가 되어야함.
#             # labelme의 Annotation을 위에서 아래로 points를 찍는 것이 아니라
#             # 아래에서 위로 찍는 경우, x1, y1이 x2, y2보다 클 수 있음.
#             scaled_bbox = [x1, y1, x2, y2]
#             stack_list.append(scaled_bbox)

#         return stack_list

    
    
    def make_scaled_bbox_list(self, bbox_list_dict):

        stack_list = []
        for bbox_key in bbox_list_dict.keys():

            bbox_list = bbox_list_dict[bbox_key]
            x1 = np.trunc(bbox_list[0][0])
            y1 = np.trunc(bbox_list[0][1])
            x2 = np.trunc(bbox_list[1][0])
            y2 = np.trunc(bbox_list[1][1])
            
            # x1과 x2중 작은 것을 쉬작 위치로 함.
            # [x_min, y_min, x_max, y_max]가 되어야함.
            # labelme의 Annotation을 위에서 아래로 points를 찍는 것이 아니라
            # 아래에서 위로 찍는 경우, x1, y1이 x2, y2보다 클 수 있음.
            if x1 <= x2:
                scaled_bbox = [x1, y1, x2, y2]
            else:
                scaled_bbox = [x2, y2, x1, y1]

            stack_list.append(scaled_bbox)

        return stack_list
    
    
    

    def make_area_list(self, bbox_list):

        stack_list = []
        for bbox in bbox_list:

            area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            stack_list.append(area)

        return stack_list
###############################################################################