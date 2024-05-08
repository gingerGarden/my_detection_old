import os
import pandas as pd

from Module.Global_variable import BASIC_KEY_DF, LABEL_DICT, INDEX_DICT_PATH, K_SIZE, SEED_LIST, TRAIN_RATIO, VALID_RATIO, SEPERATE_KEY_COLUMN_LIST, EXTERNAL, EXTERNAL_COLUMN, EXTERNAL_ELEMENT, DATASET_STRING
from Module.utils.equal_partitioning_of_a_df import seperate_random_balance_dataset
from Module.utils.Convenience_Function import save_pickle, load_pickle


DS_OBJECT = seperate_random_balance_dataset(
    key_df=BASIC_KEY_DF, key_list=SEPERATE_KEY_COLUMN_LIST,
    train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO,
    external_bool=EXTERNAL, external_col=EXTERNAL_COLUMN, external_element=EXTERNAL_ELEMENT
)


class get_index_dictionary:
    
    def __init__(self, process_boolean, path=INDEX_DICT_PATH, json_dict=LABEL_DICT, seed_list=SEED_LIST[:K_SIZE], ds_object=DS_OBJECT, dataset_string=DATASET_STRING):
        
        self.process_boolean = process_boolean
        self.path = path
        self.json_dict = json_dict
        self.img_dict_df = pd.DataFrame(json_dict['images'])
        self.seed_list = seed_list
        self.ds_object = ds_object
        self.dataset_string = dataset_string
        
        
    def process(self):

        if self.process_boolean:
            index_dict = self.index_dict_make_and_save()
            # Check if the index dictionary is correct.
            index_dictionary_checker(
                is_external=self.ds_object.external_bool, index_dict=index_dict, dataset_string=self.dataset_string
            ).check_process()
        else:
            if os.path.exists(self.path):
                index_dict = load_pickle(self.path)
            else:
                index_dict = self.index_dict_make_and_save()
                # Check if the index dictionary is correct.
                index_dictionary_checker(
                    is_external=self.ds_object.external_bool, index_dict=index_dict, dataset_string=self.dataset_string
                ).check_process()
        return index_dict
        

    def index_dict_make_and_save(self):

        index_dict = self.make_index_dict_process()
        save_pickle(index_dict, self.path)

        return index_dict

        
    def make_index_dict_process(self):

        stack_dict = dict()
        for k, seed in enumerate(self.seed_list):

            stack_dict[f"{self.dataset_string}{k}"] = self.get_one_seed_index_dictionary(seed)

        return stack_dict
        
        
    def get_one_seed_index_dictionary(self, seed):

        train_df, valid_df, test_df = self.ds_object.process(seed)
        result = {
            "train":self.choose_target_json_dictionary_from_df(df=train_df),
            "valid":self.choose_target_json_dictionary_from_df(df=valid_df),
            "test":self.choose_target_json_dictionary_from_df(df=test_df)
        }
        return result
        
        
    def choose_target_json_dictionary_from_df(self, df):

        # 대상 image들을 array로 가지고 온다.
        target_img_arr = (df["new_filekey"] + '.jpg').values
        # json_dict의 image(self.img_dict_df)에서 target_img_arr에 해당하는 records를 가지고 온다.
        target_img_df = self.img_dict_df[self.img_dict_df["file_name"].isin(target_img_arr)]
        # json_dict의 "annotation"에서 file_name에 해당하는 dictionary를 가지고 온다.
        anno_stack_dict = dict()
        for key in target_img_df['file_name'].values:
            anno_stack_dict[key] = self.json_dict['annotations'][key]
        # dictionary로 묶는다.
        result = {
            'images':target_img_df.to_dict(orient="records"),
            'annotations':anno_stack_dict
        }
        return result
    
    
    
    
class index_dictionary_checker:
    
    def __init__(self, is_external, index_dict, dataset_string):
        
        self.is_external = is_external
        self.index_dict = index_dict
        self.dataset_string = dataset_string
        
        
    def check_process(self):

        if self.is_external:
            print("This index dictionary was divided for 'External validation'.")
        else:
            print("This index dictionary was divided for 'Internal validation'.")
        print("\n")    
        self.process1_all_one_dataset_inner_check()
        print("\n")
        self.process2_all_target_dataset_outter_check(target_dataset="train")
        print("\n")
        self.process2_all_target_dataset_outter_check(target_dataset="valid")
        print("\n")
        self.process2_all_target_dataset_outter_check(target_dataset="test")
        
        
    # Process1
    ###############################################################################################
    def process1_all_one_dataset_inner_check(self):
        """
        한 Dataset 안에서 train, valid, test의 모든 file_name이 다른지 확인.
        """
        print("Process1. 모든 Dataset에 대하여 train, valid, test의 file_name들이 모두 다른지 확인.")
        print("----"*20)
        stack_list = []
        for dataset_key in self.index_dict.keys():
            point = self.one_dataset_inner_checker(dataset_key)
            stack_list.append(point)
            
        print("----"*20)
        if sum(stack_list) == 0:
            print("PASS: 모든 Dataset의 train, valid, test의 file_name은 다르다.")
        else:
            print("Warning!!!! 파일의 분할이 잘못 됐습니다!!")
        
        
        
    def one_dataset_inner_checker(self, dataset_key):

        train_df = pd.DataFrame(self.index_dict[dataset_key]['train']['images'])
        valid_df = pd.DataFrame(self.index_dict[dataset_key]['valid']['images'])
        test_df = pd.DataFrame(self.index_dict[dataset_key]['test']['images'])

        train_valid_cross = len(set(train_df['file_name']) & set(valid_df['file_name']))
        train_test_cross = len(set(train_df['file_name']) & set(test_df['file_name']))
        valid_test_cross = len(set(valid_df['file_name']) & set(test_df['file_name']))

        key = train_valid_cross + train_test_cross + valid_test_cross

        if key == 0:
            print(f"{dataset_key} - PASS: train, valid, test dataset의 모든 file_name은 다르다.")
            result = 0
        else:
            print(f"{dataset_key} - FAIL: train, valid, test dataset의 모든 file_name은 다르지 않다.")
            result = 1
        return result
    ###############################################################################################
    
    
    # Process2
    ###############################################################################################
    def process2_all_target_dataset_outter_check(self, target_dataset="train"):

        print(f"Process2. 모든 Dataset의 {target_dataset} set에 대해 filename이 중복되는지 확인.")
        print("----"*20)
        stack_list = self.compare_all_target_dataset_different(target_dataset)
        print("----"*20)
        if sum(stack_list) == 0:
            print(f"PASS: 모든 {target_dataset} dataset의 file_name은 다르다.")
        else:
            print("Warning!!!! 파일의 분할이 잘못 되었을 수 있습니다! 확인하기 바랍니다.")
    
    
    def compare_all_target_dataset_different(self, target_dataset):
        # 모든 Dataset의 train의 filename을 집합 dictionary로 가지고 온다.
        all_target_set_dict = self.get_all_dataset_target_file_name_set(target_dataset)
        # Dataset의 key들을 list로 저장
        dataset_key_list = list(all_target_set_dict.keys())
        stack_list = []
        for i in range(len(dataset_key_list) - 1):

            for dataset_key in dataset_key_list[1:]:
                pass_point = self.compare_the_target_dataset_with_other_target_dataset(
                    target_set_dict=all_target_set_dict, baseline_key=dataset_key_list[0], compare_key=dataset_key)
                stack_list.append(pass_point)

            dataset_key_list.remove(dataset_key_list[0])

        return stack_list
    
    
    
    def compare_the_target_dataset_with_other_target_dataset(self, target_set_dict, baseline_key, compare_key):

        baseline_dataset = target_set_dict[baseline_key]
        compare_dataset = target_set_dict[compare_key]

        baseline_len = len(baseline_dataset)
        compare_len = len(compare_dataset)
        intersection_len = len(baseline_dataset & compare_dataset)

        this_time_compare = f"{baseline_key} VS {compare_key}"
        if baseline_len != intersection_len:
            print(f"PASS~~~! {this_time_compare} - base:{baseline_len} / campare:{compare_len} / intersection:{intersection_len}")
            result = 0
        else:
            print(f"FAIL!!!! {this_time_compare} - base:{baseline_len} / campare:{compare_len} / intersection:{intersection_len}")
            result = 1
        return result
    
    
    
    def get_all_dataset_target_file_name_set(self, target_dataset):

        stack_dict = dict()
        for dataset_key in self.index_dict.keys():
            stack_dict[dataset_key] = set(pd.DataFrame(self.index_dict[dataset_key][target_dataset]['images'])['file_name'])

        return stack_dict
    ###############################################################################################