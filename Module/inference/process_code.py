from Module.Global_variable import torch, pd, DATASET_STRING, IMG_PARENTS_PATH
from Module.process1.index_dictionary_maker import get_index_dictionary
from Module.process1.torch_basic_style_model import get_my_torch_model
from Module.utils.Convenience_Function import get_RGB_image_by_cv2



def get_evaluate_dataset(k):
    
    idx_dict = get_index_dictionary(process_boolean=False).process()
    valid_set = idx_dict[f"{DATASET_STRING}{k}"]['valid']
    test_set = idx_dict[f"{DATASET_STRING}{k}"]['test']
    
    return valid_set, test_set



def get_model(model_key, cnn_key, parameter_path, class_num, device):
    
    model = get_my_torch_model(
        class_num=class_num,
        model_key=model_key,
        cnn_key=cnn_key
    ).process()
    model.load_state_dict(torch.load(parameter_path))
    model.to(device)
    
    return model



def get_img_info_df(label_dict):

    info_df = pd.DataFrame(label_dict['images'])
    info_df['area'] = info_df['height'] * info_df['width']

    box_size_list = []
    for idx in info_df.index:
        bbox_bool = info_df.loc[idx, "bbox"]

        if bbox_bool=='True':
            file_key = info_df.loc[idx, "file_name"]
            box_len = len(label_dict['annotations'][file_key]['bbox'])
        else:
            box_len = 0
        box_size_list.append(box_len)
    info_df['box_size'] = box_size_list
    
    return info_df



def get_info_df_img(info_df, idx, img_root=IMG_PARENTS_PATH):

    img_name = info_df.loc[idx, 'file_name']
    img_path = f"{img_root}/{img_name}"
    # cv2로 RGB 이미지를 가지고 온다.
    img = get_RGB_image_by_cv2(img_path, RGB=True)
    
    return img