import torch
import numpy as np



# torch style로 기본적인 전처리 실시
def torch_image_transform(img, float_type=torch.float32):
    
    img = img/255
    torch_img = torch.as_tensor(img, dtype=float_type)
    torch_img = torch.permute(torch_img, (2, 0, 1))
    
    return torch_img



def Tensor_RGB_img_convert_to_numpy_style(tensor_img:torch.Tensor):
    """
    Tensor 이미지(channel, height, width)를
    Numpy 이미지(height, width, channel)로 변환한다.
    """
    np_img = tensor_img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    
    return np_img
    


def state_dict_to_np_array_dict(params_state_dict):
    """
    Tensor(cuda)로 구성된 state_dict을 numpy array로 만들어서 dictionary에 담는다.
    """
    result = dict()
    for param_key in params_state_dict.keys():
        result[param_key] = params_state_dict[param_key].cpu().numpy()
    return result


def image_list_upload_to_device(img_list, device):
    """
    image들이 list로 묶여 있는 상태에서 이들을 모두 device에 upload 한다.
    """
    # mini-batch의 image들은 list로 묶여 있음. 각각 .to(device) 정의
    return [img.to(device) for img in img_list]



def target_list_upload_to_device(target_list, device):
    """
    object detection에서 target이 torch.Tensor인 경우 device에 올린다.
    """
    result = []
    for target in target_list:
        device_dict = dict()
        for key, value in target.items():
            device_dict[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        result.append(device_dict)
    return result



class torch_device:
    
    def __init__(self):
        self.gpu_key = torch.backends.cudnn.is_available()
        self.gpu_count = torch.cuda.device_count()
        
        
    def gpu_check(self):
        print("Pytorch GPU check")
        print("torch version: ", torch.__version__)
        print("===="*15)
        print(f"GPU use: {self.gpu_key}")
        if self.gpu_key:
            print(f"GPU count: {self.gpu_count}")
            for i in range(self.gpu_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"use CPU")
        print("===="*15)
        
        
    def get_device(self, gpu_number):
        if self.gpu_key:
            device = self.check_gpu_number_contain_having(gpu_number)
        else:
            device = 'cpu'
        return device
            
            
    def check_gpu_number_contain_having(self, gpu_number):
        my_gpu_max_number = self.gpu_count - 1
        if gpu_number > my_gpu_max_number:
            print(f"Warning! You have {self.gpu_count} GPUs, and the maximum selectable number is {my_gpu_max_number}!")
            print("Since it's not a selectable GPU number, returning the default GPU 0 as the device.")
            device = 'cuda:0'
        else:
            device = f'cuda:{gpu_number}'
        return device