from Module.Global_variable import torch, np, cv2, DATASET_STRING, IMG_PARENTS_PATH, BATCH_SIZE, WORKER_NUM, AUG_Ins
from Module.utils.Convenience_Function import get_RGB_image_by_cv2




class my_object_detect_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dict, img_root, augment=None, dtype=torch.float32):
        self.img_dict = data_dict['images']
        self.anno_dict = data_dict['annotations']
        self.img_root = img_root
        self.augment = augment
        self.dtype = dtype
        
        
    def __len__(self):
        return len(self.img_dict)
    
    
    def __getitem__(self, idx):
        
        # image를 가지고 온다.
        idx_img_dict = self.img_dict[idx]
        image = self.get_image(idx_img_dict)
        
        # target(label) 관련 정보를 가지고 온다.
        target = self.get_target(idx_img_dict)
        
        # 데이터 증강 여부
        if self.augment is not None:
            image, target = self.augment.process(image, target)
        
        # Torch의 학습 방식에 맞게 가장 기본적인 전처리 실시.
        torch_image = self.img_basic_transform(image)
        torch_target = self.target_basic_transform(target)
        
        return torch_image, torch_target
        
        
    def get_image(self, idx_img_dict):

        img_name = idx_img_dict['file_name']
        img_path = f"{self.img_root}/{img_name}"
        img = get_RGB_image_by_cv2(img_path, RGB=True)
        
        return img
    
    
    def get_target(self, idx_img_dict):

        idx_anno_dict = self.anno_dict[idx_img_dict['file_name']]
        target = {
            "boxes":idx_anno_dict['bbox'],
            "labels":idx_anno_dict['category_id'],
            "image_id":idx_img_dict['id'],
            "area":idx_anno_dict['area'],
            "iscrowd":idx_anno_dict['iscrowd']
        }
        return target
    
    
    def img_basic_transform(self, img):
        
        # torch에 맞게 특성 수정
        img = img/255
        torch_img = torch.as_tensor(img, dtype=self.dtype)
        torch_img = torch.permute(torch_img, (2, 0, 1))
#         img = np.transpose(img_array, (2,0,1))

        return torch_img
    
    
    def target_basic_transform(self, target):
        new_target = dict()
        if len(target["boxes"]) == 0:
            new_target["boxes"] = torch.as_tensor(target['boxes'], dtype=self.dtype).reshape(0, 4)
        else:
            new_target["boxes"] = torch.as_tensor(target['boxes'], dtype=self.dtype)
            
        new_target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        new_target["image_id"] = target["image_id"]
        new_target["area"] = torch.as_tensor(target["area"], dtype=torch.int64)
        new_target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
        
        return new_target
        
        
    
    
    
    
def collate_fn_for_OD(batch):
    """
    Object detection은 입력되는 이미지의 크기가 다른 것을 상정하므로, 해당 코드를 통해 다른 크기의 Tensor를
    Data Loader가 Batch로 출력할 수 있게 함(DataLoader는 본래 같은 크기의 Tensor들을 Batch로 출력).
    
    * `*batch`: 반복 가능한 객체를 unpacking 함. `'*batch`는 `batch`내 각 요소를 별도의 인자로 함수에 전달.
    * `zip(*batch)`: `zip` 함수는 여러 반복 가능한 객체의 요소를 짝지어 Tuple로 묶음.
    * `tuple(zip(*batch))`: `zip`에 의해 생성된 Tuple들의 Sequence를 `tuple` 함수로 다시 Tuple로 변환.
       위 방법을 쓰면 `batch`의 구조가 전치(Transpose)되어 원래의 각 열이 행으로, 각 행이 열로 변함.
    """
    return tuple(zip(*batch))

    
    
class get_my_dataLoader:
    
    def __init__(
        self, idx_dict,
        img_root=IMG_PARENTS_PATH, data_key=DATASET_STRING, batch_size=BATCH_SIZE, num_worker=WORKER_NUM
    ):
        self.idx_dict = idx_dict
        self.img_root = img_root
        self.data_key = data_key
        self.batch_size = batch_size
        self.num_worker = num_worker
        
        
        
    def get_all_torch_dataLoader(self, k):
        
        train_dataset, valid_dataset, test_dataset = self.get_all_torch_dataset(k)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            pin_memory=torch.cuda.is_available(),
#             drop_last=True,
            num_workers=self.num_worker,
            collate_fn=collate_fn_for_OD
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = self.batch_size,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_worker,
            collate_fn=collate_fn_for_OD
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = self.batch_size,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_worker,
            collate_fn=collate_fn_for_OD
        )
        return train_loader, valid_loader, test_loader
        
        
        
    def get_all_torch_dataset(self, k):

        target_idx_dict = self.idx_dict[f"{self.data_key}{k}"]

        train_dataset = my_object_detect_Dataset(
            data_dict=target_idx_dict["train"],
            img_root=self.img_root,
            augment=AUG_Ins
        )
        valid_dataset = my_object_detect_Dataset(
            data_dict=target_idx_dict["valid"],
            img_root=self.img_root,
            augment=None
        )
        test_dataset = my_object_detect_Dataset(
            data_dict=target_idx_dict["test"],
            img_root=self.img_root,
            augment=None
        )
        return train_dataset, valid_dataset, test_dataset