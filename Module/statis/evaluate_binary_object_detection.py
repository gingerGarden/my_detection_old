import pandas as pd
import numpy as np

from copy import deepcopy
from sklearn import metrics

from Module.Global_variable import IMG_PARENTS_PATH
from Module.utils.Convenience_Function import get_RGB_image_by_cv2, draw_img_and_bbox_torch_style
from Module.statis.my_stats import calculate_iou, calculate_f1_score




class Evaluate_Object_Detection:
    
    def __init__(self, label_dict, predict_dict, label_bbox_key="bbox", pred_bbox_key="boxes", conf_key="scores", img_root=IMG_PARENTS_PATH):
        
        self.label_dict = label_dict
        self.predict_dict = predict_dict
        self.label_bbox_key = label_bbox_key
        self.pred_bbox_key = pred_bbox_key
        self.conf_key = conf_key
        self.img_root = img_root
        
        self.iou_df = None
        self.GT_size = None
        
        
    # 주요 기능1) iou_df, GT_size 계산
    def get_GT_number_and_iou_df(self):

        iou_df = pd.DataFrame()
        GT_size = 0

        for img_id, predict in self.predict_dict.items():

            label = self.get_one_predict_label(img_id)

            # sub_iou_df와 sub_GT의 갯수를 가지고 온다.
            sub_iou_df, sub_GT_size = make_one_record_iou_df(
                GT_boxes=label[self.label_bbox_key],
                Pred_boxes=predict[self.pred_bbox_key],
                Conf_score=predict[self.conf_key]
            ).process()
            
            sub_iou_df["img_id"] = img_id
            iou_df = pd.concat([iou_df, sub_iou_df])
            GT_size += sub_GT_size

        self.iou_df = iou_df.reset_index(drop=True)
        self.GT_size = GT_size
        
        
    # 주요 기능2) 평가 지표 계산
    def calculate_process(self):
        """
        COCO Challenge는 다음과 같이 표기한다.
        ----------------------------------------------------------------
        AP: AP@[.50:.05:0.95]와 같다.
        AP50: IoU의 threshold를 0.50로 하였을 때, AP다.
        AP75: IoU의 threshold를 0.75로 하였을 때, AP다.
        precision, recall, f1-score는 모두 0.50일때의 것이다.
        """
        # IoU DataFrame과 GT_number를 가지고 온다.
        self.get_GT_number_and_iou_df()
        # Confidence score를 기준으로 내림차순 정렬 한다.
        self.iou_df = self.iou_df.sort_values("confidence", ascending=False).reset_index(drop=True)

        # 모델 평가를 위한 지표 계산
        ## AP@.50
        score_dict_50 = self.get_the_score_dict_for_a_specific_IoU_threshold(threshold=0.5)
        ## AP@.75
        score_dict_75 = self.get_the_score_dict_for_a_specific_IoU_threshold(threshold=0.75)
        ## AP(AP@[.50:05:95]) 계산
        AP = self.calculate_AP50_05_95()
        result = {
            "AP(AP@[.50:.05:.95])":AP,
            "AP50":score_dict_50['mAP'],
            "AP75":score_dict_75['mAP'],
            "precision":score_dict_50["precision"],
            "recall":score_dict_50["recall"],
            "f1_score":score_dict_50["f1_score"]
        }
        return result
    
        
        
    # 특정 IoU threshold를 기준으로 모델 평가 지표 계산
    def get_the_score_dict_for_a_specific_IoU_threshold(self, threshold):

        score_dict = calculate_score_dict(
            iou_df = self.iou_df,
            gt_size = self.GT_size,
            threshold = threshold
        ).process()

        return score_dict
    
    
    
    def calculate_AP50_05_95(self):

        mAP_list = []
        for now_threshold in np.arange(0.5, 1.0, 0.05):

            score_dict = self.get_the_score_dict_for_a_specific_IoU_threshold(threshold=now_threshold)
            mAP_list.append(score_dict['mAP'])

        result = np.mean(mAP_list)

        return result
        
        
        
    # predict된 mini-batch 중 하나의 label을 가지고 온다.
    def get_one_predict_label(self, img_id):
        label = self.label_dict['annotation'][
            self.label_dict['id_image'][img_id]['file_name']
        ]
        return label
    
    
    
    # img_id에 해당하는 predict와 label의 bbox를 가지고 온다.
    def get_target_id_pred_and_label(self, img_id):

        predict = self.predict_dict[img_id]
        label = self.get_one_predict_label(img_id)

        return predict, label
    
    
    
    # 모델이 추론한 결과를 시각화함.
    def predicted_bboxes_visualization(self, img_id):

        # get image
        img_key = self.label_dict['id_image'][img_id]['file_name']
        img_path = f"{self.img_root}/{img_key}"
        img = get_RGB_image_by_cv2(img_path, RGB=True)
        # get predicted bboxes
        pred_bboxes = self.predict_dict[img_id][self.pred_bbox_key]

        draw_img_and_bbox_torch_style(img, bbox_list=pred_bboxes, title=img_key)


        

class calculate_score_dict:
    
    def __init__(self, iou_df, gt_size, threshold=0.5):
        
        self.iou_df = deepcopy(iou_df)
        self.gt_size = gt_size
        self.threshold = threshold
        
        
        
    def process(self):

        # 계산을 위한 table 생성
        self.Complete_the_table()
        # score 생성
        score_dict = self.calculate_evaluate_scores()
        
        return score_dict
        
        
        
    # 지표 게산이 가능하도록 Table에 필요한 컬럼들을 추가한다. 
    def Complete_the_table(self):

        # threshold에 맞춰서 TP와 FP를 정의
        self.iou_df['TPnFP'] = np.where(self.iou_df['iou'] >= self.threshold, "TP", "FP")
        # Acc_TP와 Acc_FP 컬럼 추가
        self.add_Acc_TP_and_ACC_FP_columns()
        # precision 계산
        self.iou_df["precision"] = self.iou_df["AccTP"]/(self.iou_df["AccTP"] + self.iou_df["AccFP"])
        # recall 계산
        self.iou_df["recall"] = self.iou_df["AccTP"]/self.gt_size
        
        

    # 성능을 평가할 precision, recall, f1-score, mAP를 가지고 온다.
    def calculate_evaluate_scores(self):
        # mAP는 class들의 mean인데, class가 1개이므로, AP는 mAP다.
        precision = self.iou_df.tail(1)['precision'].item()
        recall = self.iou_df.tail(1)['recall'].item()
        fl_score = calculate_f1_score(precision, recall)
        mAP = metrics.auc(self.iou_df["recall"], self.iou_df["precision"])
        
        result = {
            "precision":precision,
            "recall":recall,
            "f1_score":fl_score,
            "mAP":mAP
        }
        return result
        
        
        
    # Acc_TP, Acc_FP 컬럼 추가
    def add_Acc_TP_and_ACC_FP_columns(self):
        TP_stack = 0
        FP_stack = 0
        stack_list = []
        for idx in self.iou_df.index:

            TFPN = self.iou_df.loc[idx, "TPnFP"]
            if TFPN == "TP":
                TP_stack += 1
            else:
                FP_stack += 1
            record = (TP_stack, FP_stack)
            stack_list.append(record)

        Acc_TFPN_df = pd.DataFrame(stack_list, columns=["AccTP", "AccFP"])
        self.iou_df = pd.merge(self.iou_df, Acc_TFPN_df, left_index=True, right_index=True)
        
        
        
    # precison-recall curve를 그린다.
    def draw_precision_recall_curve(self, _figSize=(12, 8), draw_all_lim=False):

        title_font = {
            'family':'serif',
            'color':'black',
            'weight':'bold',
            'size':20
        }
        xylabel_font = {
            'family':'serif',
            'color':'black',
            'size':15
        }

        plt.figure(figsize=_figSize)
        plt.plot(self.iou_df["recall"], self.iou_df["precision"])
        
        if draw_all_lim:
            plt.xlim(0, 1)
            plt.ylim(0, 1)

        plt.title("Precision-Recall Curve", fontdict=title_font, pad=20)
        plt.xlabel("Recall", fontdict=xylabel_font, labelpad=10)
        plt.ylabel("Precision", fontdict=xylabel_font, labelpad=10)

        plt.show()
        
        


class make_one_record_iou_df:
    
    def __init__(self, GT_boxes, Pred_boxes, Conf_score, iou_cut=0.01):
        
        self.GT_boxes = GT_boxes
        self.Pred_boxes = Pred_boxes
        self.Conf_score = Conf_score
        self.iou_cut = iou_cut
        
        self.column_list = ["ground_truth", 'predict', "iou", "confidence"]
        self.None_df = pd.DataFrame(columns=self.column_list)
        
        self.iou_df = None
        
        
        
    def process(self):
        """
        * FN 계산
          - FN은 다음 두 가지 경우에 의해 발생한다.
        ---------------------------------------------------------------------
        1) self.GT_boxes가 존재하나, self.Pred_boxes가 없는 경우.
        2) self.GT_boxes가 존재하고, self.Pred_boxes가 부족한 경우.
        ---------------------------------------------------------------------
        2)의 경우, 중복된 predicted bbox의 수로 인해 파악이 어렵다.
          > 그러므로, iou가 iou_cut 이하인, 절대 GT가 아닐 것으로 파악되는 record를
            제외하고 중복되지 않은 GT의 수를 세어, 찾지 못한 GT의 수를 계산하고,
            여기에 중복이 허용된 "절대 GT가 아닐 것으로 파악되는 record"의 크기를
            더 하여, GT의 크기를 정의한다.
        """
        # GT boxes는 존재하나, Pred boxes가 없는 경우(예측 실패) - GT_bboxes 크기만큼 GT_num 추가
        if (len(self.GT_boxes) > 0) & (len(self.Pred_boxes) == 0):
            iou_df = self.None_df
            GT_size = len(self.GT_boxes)
        # GT boxes가 존재하지 않으나, Pred boxes가 있는 경우(없는 대상을 예측) - IoU가 0인 record 추가
        elif (len(self.GT_boxes) == 0) & (len(self.Pred_boxes) > 0):
            self.make_only_FP_iou_df()
            iou_df = self.iou_df
            GT_size = 0
        # 일반적인 Case
        else:
            # label과 predicted의 모든 IoU의 경우의 수 계산
            self.calculate_all_IoU()
            # predict bbox를 기준으로 가장 IoU가 큰 record만 가지고 온다.
            self.Select_the_predicted_bbox_with_the_highest_IoU()
            iou_df = self.iou_df
            # 위 과정만으로 FP와 TP의 크기는 예상할 수 있으나, FN의 크기는 예상할 수 없다.
            GT_size = self.find_duplicated_GT_size()
            
        return iou_df, GT_size
            
            
        
    # predict된 모든 bbox와 GT의 모든 bbox의 IoU 계산
    def calculate_all_IoU(self):

        stack_list = []
        for i, gt_bbox in enumerate(self.GT_boxes):

            for j, pred_bbox in enumerate(self.Pred_boxes):

                iou = calculate_iou(gt_bbox, pred_bbox)
                conf_score = self.Conf_score[j]
                record = (i, j, iou, conf_score)
                stack_list.append(record)  
        self.iou_df = pd.DataFrame(stack_list, columns=self.column_list)
        
        
        
    def Select_the_predicted_bbox_with_the_highest_IoU(self):

        # predict의 key를 unique 하게 가지고 온다.
        pred_unique_key = self.iou_df["predict"].unique()

        # predict box를 주인공으로 하여, 각 predict 중 iou가 가장 높은 record만 가지고 온다.
        stack_list = []
        for pred_key in pred_unique_key:

            one_pred_df = self.iou_df[self.iou_df["predict"] == pred_key]
            target_idx = one_pred_df[one_pred_df["iou"] == one_pred_df['iou'].max()].index[0]

            stack_list.append(target_idx)

        self.iou_df = self.iou_df.loc[stack_list].reset_index(drop=True)
        
        
        
    def find_duplicated_GT_size(self):
        """
        GT의 크기 계산.
        -----------------------------------------------------------------------------
        모델이 찾아낸 GT는 중복되어 있을 수 있으며, 이는 unique한 GT의 크기 비교가 불가능.
        - 만약, 모델이 찾아낸 GT의 unique한 크기보다 실제 GT의 크기가 크다면 다음 프로세스 진행
          1) 모델이 찾아낸 GT의 unique한 크기 계산(find_unique_GT_size).
          2) 실제 GT의 크기(real_GT_size)와 find_unique_GT_size의 차이 계산(not_found_GT_size)
          3) 모델이 찾아낸 GT의 길이(중복허용)에 not_found_GT_size 합.

        - 만약, 모델이 찾아낸 GT의 unique한 크기와 실제 GT의 크기가 같다면 GT_df의 길이를 내보낸다.
        """
        # 확실하게 FP가 아닌 GT일 수 있는 대상들만 filtering
        GT_df = self.iou_df[self.iou_df["iou"] > self.iou_cut]
        # 모델이 찾아낸 unique한 GT의 크기
        find_unique_GT_size = len(GT_df["ground_truth"].unique())
        # 실제 unique한 GT의 크기
        real_GT_size = len(self.GT_boxes)

        # 실제 unique한 GT의 크기보다 찾아낸 unique한 GT의 크기가 작은 경우.
        if  real_GT_size > find_unique_GT_size:
            # 모델이 찾아내지 못한 unique한 GT의 크기
            not_found_GT_size = real_GT_size - find_unique_GT_size
            GT_size = not_found_GT_size + len(GT_df)
        # 그렇지 않은 경우엔, 중복을 고려하여 GT_df의 길이를 반환한다.
        else:
            GT_size = len(GT_df)

        return GT_size
    
        
        
    def make_only_FP_iou_df(self):
        
        stack_list = []
        for i in range(len(self.Pred_boxes)):
            conf_score = self.Conf_score[i]
            record = (None, i, 0, conf_score)
            stack_list.append(record)
        
        self.iou_df = pd.DataFrame(stack_list, columns=self.column_list)