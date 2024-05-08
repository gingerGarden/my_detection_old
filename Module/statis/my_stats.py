import numpy as np
from sklearn import metrics



# iou 계산
def calculate_iou(bbox1, bbox2):

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    # 교집합 영역
    inter_area = max(x2-x1, 0) * max(y2-y1, 0)
    intersection_area = inter_area if inter_area > 0 else 0
    # 합집합의 영역 계산
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    # IoU 계산
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou



# NMS 계산
def None_Maximum_Suppression(boxes, scores, iou_threshold=0.5):
    """
    NMS(None-Max Suppression) 의 흐름
    ----------------------------------------------------------------
    Step 1) Confidence score에 대하여 내림차순하여, 가장 높은 bbox 선택
    Step 2) 선택된 bbox와 다른 모든 bbox의 IoU 비교
    Step 3) 다른 bbox와 IoU가 iou_threshold 이상이면 목록에서 제거
    Step 4) 반복
    ----------------------------------------------------------------
    Output
    boxes[keep_idx]: 생존한 bboxes
    scores[keep_idx]: 생존한 bboxes의 Confidence score
    keep_idx: 생존한 bboxes의 Confidence score의 index
    """
    # 내림차순 정렬된 Confidence score의 index
    idx_arr = np.argsort(scores)[::-1]
    keep_idx = []
    # Step4. 반복
    while len(idx_arr) > 0:
        # Step1. 목록에서 가장 높은 Confidence box는 생존한다.
        current_index = idx_arr[0]
        keep_idx.append(current_index)

        # Step2. 다른 box들과 IoU 비교
        save_idx = []
        current_box = boxes[current_index]

        for now_idx in idx_arr[1:]:
            now_box = boxes[now_idx]
            iou = calculate_iou(bbox1=current_box, bbox2=now_box)

            # Step3. iou가 iou_threshold 미만이면 생존
            if iou < iou_threshold:
                save_idx.append(now_idx)

        idx_arr = save_idx
    
    return boxes[keep_idx], scores[keep_idx], keep_idx



# Precision 계산
def calculate_precision(TP, FP):
    denominator = (TP + FP)
    if denominator > 0:
        precision = TP/denominator
    else:
        precision = 0
    return precision



# Recall 계산
def calculate_recall(TP, FN):
    denominator = (TP + FN)
    if denominator > 0:
        recall = TP/denominator
    else:
        recall = 0
    return recall



# Precision과 Recall의 조화 평균
def calculate_f1_score(precision, recall):
    denominator = (precision + recall)
    if denominator > 0:
        f1_score = 2*(precision*recall)/denominator
    else:
        f1_score = 0
    return f1_score