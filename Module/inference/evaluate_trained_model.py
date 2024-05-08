from Module.Global_variable import torch, np, pd, IMG_PARENTS_PATH, WORKER_NUM
from Module.process1.torch_dataset import my_object_detect_Dataset
from Module.utils.log_utils import my_time_log_progressbar
from Module.utils.Convenience_Function_by_torch import image_list_upload_to_device, target_list_upload_to_device
from Module.utils.Convenience_Function import show_markdown_df
from Module.statis.evaluate_binary_object_detection import Evaluate_Object_Detection
from Module.statis.my_stats import None_Maximum_Suppression



class model_performance_check:
    
    def __init__(self, model, valid_dict, test_dict, batch_size, device, img_root=IMG_PARENTS_PATH, worker=WORKER_NUM):
        
        self.model = model
        self.valid_dict = valid_dict
        self.test_dict = test_dict
        self.batch_size = batch_size
        self.device = device
        self.img_root = img_root
        self.worker = worker
        
        
    def process(self):
        
        valid_loader = self.get_data_loader(data_dict=self.valid_dict)
        test_loader = self.get_data_loader(data_dict=self.test_dict)
        # 모델 추론
        evaluate_ins = evaluate_model(model=self.model, device=self.device)
        valid_pred_dict = evaluate_ins(valid_loader)
        test_pred_dict = evaluate_ins(test_loader)
        # 추론 결과 요약
        # label 파일을 추론 형태로 변환
        valid_eval_dict = self.label_dict_transrate_to_evaluate_style(label_dict=self.valid_dict)
        test_eval_dict = self.label_dict_transrate_to_evaluate_style(label_dict=self.test_dict)
        valid_score = Evaluate_Object_Detection(label_dict=valid_eval_dict, predict_dict=valid_pred_dict).calculate_process()
        test_score = Evaluate_Object_Detection(label_dict=test_eval_dict, predict_dict=test_pred_dict).calculate_process()
        score_df = self.show_model_performance(valid_score, test_score)
        
        return score_df
        
        
    def get_data_loader(self, data_dict):

        def collate_fn_for_OD(batch):
            return tuple(zip(*batch))

        dataset = my_object_detect_Dataset(
            data_dict=data_dict,
            img_root=self.img_root,
            augment=None
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.worker,
            collate_fn=collate_fn_for_OD
        )
        return data_loader
    
    
    def label_dict_transrate_to_evaluate_style(self, label_dict):

        result = dict()
        result['id_image'] = pd.DataFrame(label_dict['images']).set_index('id').to_dict(orient='index')
        result['annotation'] = label_dict['annotations']

        return result
    
    
    def show_model_performance(self, valid_score, test_score):

        valid_df = pd.DataFrame([(key, item) for key, item in valid_score.items()], columns=['score', 'valid'])
        test_df = pd.DataFrame([(key, item) for key, item in test_score.items()], columns=['score', 'test'])
        result = pd.merge(valid_df, test_df, on='score')
        result.set_index('score', inplace=True)
        show_markdown_df(result.round(5))

        return result

    

class evaluate_model:
    
    def __init__(self, model, device, iou_threshold=0.3, cs_threshold=0.5):
        
        self.model = model
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.iou_threshold = iou_threshold
        self.cs_threshold = cs_threshold
        
        

    @torch.inference_mode()
    def __call__(self, data_loader):

        self.model.eval()
        n_threads = torch.get_num_threads()   # torch가 현재 사용 중인 Thread 수 반환
        torch.set_num_threads(1)              # 사용할 Thread 수를 한 개로 제한

        pb_ins = my_time_log_progressbar(header="evaluate: ", num_size=3)
        predict_dict = dict()
        for imgs, targets in pb_ins.with_time_log(data_loader):

            imgs = image_list_upload_to_device(img_list=imgs, device=self.device)
            targets = target_list_upload_to_device(target_list=targets, device=self.device)
            outputs = self.predict_synchronize_model(imgs, targets)
            predict_dict = predict_dict | outputs
        
        self.apply_NMS_to_predict_dict(predict_dict)                 # NMS 적용
        self.remove_under_Confidence_Score_threshold(predict_dict)   # Confidence score에 대한 Threshold 적용
        
        torch.set_num_threads(n_threads)
        return predict_dict
    
            
            
    def predict_synchronize_model(self, imgs, targets):
        # GPU가 사용 가능한 경우, GPU에서 백그라운드로 돌아가는 연산이 완료될 때까지 기다림.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # 추론
        with torch.no_grad():
            outputs = self.model(imgs)
        outputs = [{k: v.to(self.cpu_device).detach().numpy() for k, v in t.items()} for t in outputs] # gpu의 tensor에서 cpu의 numpy로 변환
        outputs = {target["image_id"]: output for target, output in zip(targets, outputs)}   # image_id를 붙여서 내보낸다.
        return outputs
    
    
    
    def apply_NMS_to_predict_dict(self, predict_dict):

        for key, value in predict_dict.items():

            boxes=value['boxes']
            scores=value['scores']
            labels=value['labels']

            if len(boxes) > 0:
                new_boxes, new_scores, score_idx = None_Maximum_Suppression(
                    boxes, scores, iou_threshold=self.iou_threshold
                )
                new_labels = labels[score_idx]

                value['boxes'] = new_boxes
                value['scores'] = new_scores
                value['labels'] = new_labels
    
    

    # bbox_threshold 보다 낮은 bbox는 제거한다.
    def remove_under_Confidence_Score_threshold(self, predict_dict):
        for key, item  in predict_dict.items():
            if len(item['boxes']) == 0:
                pass
            else:
                predict_dict[key] = self.check_one_predict(a_predict=item)
        
        
        
    # 하나의 추론 결과 확인.
    def check_one_predict(self, a_predict):
        boxes_list = []
        labels_list = []
        scores_list = []

        for i in range(len(a_predict['scores'])):
            box = a_predict['boxes'][i]
            label = a_predict['labels'][i]
            score = a_predict['scores'][i]
            if score >= self.cs_threshold:
                boxes_list.append(box)
                labels_list.append(label)
                scores_list.append(score)
                
        result = {
            "boxes":np.array(boxes_list),
            "labels":np.array(labels_list),
            "scores":np.array(scores_list)
        }

        return result