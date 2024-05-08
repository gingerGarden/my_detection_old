from Module.Global_variable import torch, np

from Module.statis.my_stats import None_Maximum_Suppression
from Module.utils.Convenience_Function_by_torch import image_list_upload_to_device, torch_image_transform




class inference:
    
    def __init__(self, model, device, iou_threshold=0.3, cs_threshold=0.5):
        """
        한 이미지에 대해 추론한다.
        """
        self.model = model
        self.device = device
        self.iou_threshold = iou_threshold
        self.cs_threshold = cs_threshold
        
        
    def __call__(self, img):
        # torch 스타일로 변환
        torch_img = [torch_image_transform(img)]
        # 장치에 upload
        torch_img = image_list_upload_to_device(torch_img, device=self.device)
        # 모델로 추론
        output = self.model_inference(img_list=torch_img)
        # NMS 적용
        self.apply_NMS_to_predict_dict(output)                 # NMS 적용
        self.remove_under_Confidence_Score_threshold(output)   # Confidence score에 대한 Threshold 적용

        return output[0]
        
        
    @torch.inference_mode()
    def model_inference(self, img_list):

        cpu_device = torch.device("cpu")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_list)
        outputs = {0:{key:value.to(cpu_device).detach().numpy() for key, value in outputs[0].items()}}

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