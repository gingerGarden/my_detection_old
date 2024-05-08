from Module.Global_variable import time, torch, np

from Module.statis.my_stats import None_Maximum_Suppression
from Module.statis.evaluate_binary_object_detection import Evaluate_Object_Detection
from Module.utils.Convenience_Function_by_torch import image_list_upload_to_device, target_list_upload_to_device





class model_evaluate:
    
    def __init__(self, model, valid_Loader, test_Loader, device, CS_threshold, iou_threshold, label_idx_dict, log_ins, verbose):
        
        self.model = model
        self.valid_Loader = valid_Loader
        self.test_Loader = test_Loader
        self.device = device
        self.CS_threshold = CS_threshold
        self.iou_threshold = iou_threshold
        self.label_idx_dict = label_idx_dict
        self.log_ins = log_ins
        self.verbose = verbose
        
        self.cpu_device = torch.device("cpu")
        
        
        
    @torch.inference_mode()
    def valid_one_epoch(self, epoch):

        start_time = time.time()

        # 추론에 사용되는 Thread 수 제한
        n_threads = torch.get_num_threads()   # torch가 현재 사용 중인 Thread 수 반환
        torch.set_num_threads(1)              # 사용할 Thread 수를 한 개로 제한

        # Log instance를 가지고 온다.
        Time_Log_Ins, Loss_Log_Ins = self.log_ins.get_log_instances(epoch, key="valid")

        # Validation set에 대한 추론
        predict_dict = dict()
        for imgs, targets in Time_Log_Ins.with_time_log(self.valid_Loader):

            imgs = image_list_upload_to_device(img_list=imgs, device=self.device)
            targets = target_list_upload_to_device(target_list=targets, device=self.device)
            # Model 추론
            outputs = self.predict_synchronize_model(imgs, targets)
            predict_dict = predict_dict | outputs
            # Loss 계산
            loss_dict = self.get_eval_model_loss_dict(imgs, targets)
            Loss_Log_Ins.add_loss_dict(loss_dict)   # stack Loss Log

        
        self.apply_NMS_to_predict_dict(predict_dict)                 # NMS 적용
        self.remove_under_Confidence_Score_threshold(predict_dict)   # Confidence score에 대한 Threshold 적용

        # thread 수를 초기 상태로 조정
        torch.set_num_threads(n_threads)

        # 모델 성능 평가
        score_dict = Evaluate_Object_Detection(label_dict=self.label_idx_dict, predict_dict=predict_dict).calculate_process()

        # Log 생성
        mean_loss_dict = Loss_Log_Ins.make_mean_loss_dict()
        if self.verbose:
            Loss_Log_Ins.print_typical_log_sentence(start_time, loss_dict=mean_loss_dict, score_dict=score_dict)

        return mean_loss_dict, score_dict
        
        
        
    @torch.inference_mode()
    def model_test(self, epoch):

        start_time = time.time()

        # 추론에 사용되는 Thread 수 제한
        n_threads = torch.get_num_threads()   # torch가 현재 사용 중인 Thread 수 반환
        torch.set_num_threads(1)              # 사용할 Thread 수를 한 개로 제한

        # Log instance를 가지고 온다.
        Time_Log_Ins, Loss_Log_Ins = self.log_ins.get_log_instances(epoch, key="test")

        # Validation set에 대한 추론
        predict_dict = dict()
        for imgs, targets in Time_Log_Ins.with_time_log(self.test_Loader):

            imgs = image_list_upload_to_device(img_list=imgs, device=self.device)
            targets = target_list_upload_to_device(target_list=targets, device=self.device)
            # Model 추론
            outputs = self.predict_synchronize_model(imgs, targets)
            predict_dict = predict_dict | outputs

        self.apply_NMS_to_predict_dict(predict_dict)                 # NMS 적용
        self.remove_under_Confidence_Score_threshold(predict_dict)   # Confidence score에 대한 Threshold 적용
        score_dict = Evaluate_Object_Detection(
            label_dict=self.label_idx_dict, predict_dict=predict_dict
        ).calculate_process()    # 모델 성능 평가

        # thread 수를 초기 상태로 조정
        torch.set_num_threads(n_threads)

        # Log 생성
        if self.verbose:
            Loss_Log_Ins.print_test_score_sentence(start_time, score_dict)

        return predict_dict, score_dict
        
        
        
    # 동기화된 모델을 통해서 image에 대해 추론함.
    def predict_synchronize_model(self, imgs, targets):
        """
        `torch.cuda.synchronize()`
        : torch에서 GPU 연산과 관련된 모든 CUDA 스트림 동기화.
        >>> torch에서 cuda 연산은 기본적으로 비동기적으로 수행되어, CPU 코드가 실행되는 동안 별도의 GPU 연산이 백그라운드에서 동시에 수행됨.
        >>> 동기화: `torch.cuda.synchronize()`는 호출된 시점에서 GPU에 대기 중인 모든 작업이 완료될 때까지 CPU 실행 차단
            - CPU 코드는 GPU 연산이 완료될 때까지 대기 상태.
        >>> 디버깅, 정확한 성능 측정, 여러 GPU를 사용하는 복잡한 시나리오에서 GPU간 연산이 서로에게 영향을 주지 않도록 동기화하는 데 사용.
            - 동기화는 성능에 영향을 줄 수 있으므로, 디버깅, 성능 측정 목적이 아닌 경우, 사용을 최소화하는 것이 좋음.
        """
        # GPU가 사용 가능한 경우, GPU에서 백그라운드로 돌아가는 연산이 완료될 때까지 기다림.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # 추론
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(imgs)
        outputs = [{k: v.to(self.cpu_device).detach().numpy() for k, v in t.items()} for t in outputs] # gpu의 tensor에서 cpu의 numpy로 변환
        outputs = {target["image_id"]: output for target, output in zip(targets, outputs)}   # image_id를 붙여서 내보낸다.
        return outputs
        
        
        
    # 추론 상태의 모델로부터 loss_dict을 출력함.
    def get_eval_model_loss_dict(self, imgs, targets):
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(imgs, targets)
        return loss_dict

    
    
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
            if score >= self.CS_threshold:
                boxes_list.append(box)
                labels_list.append(label)
                scores_list.append(score)
                
        result = {
            "boxes":np.array(boxes_list),
            "labels":np.array(labels_list),
            "scores":np.array(scores_list)
        }

        return result
    