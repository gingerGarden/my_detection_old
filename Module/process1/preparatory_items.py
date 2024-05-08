import numpy as np

from Module.Global_variable import (
    NUMBER_OF_CLASS, EPOCHS, GRADIENT_CLIIPING_MAX_NORM, EARLY_STOPPING, PATIENCE, USE_AMP, K_SIZE,
    LOG_DIR, ITERATION_LOG_DIR, ITER_LOG_KEY, TRAINING_LOG_DIR, MODEL_DIR
)
from Module.utils.Convenience_Function import new_dir_maker



def make_process_start_dir(makes_new=False):
    
    new_dir_maker(LOG_DIR, makes_new)
    new_dir_maker(f"{LOG_DIR}/{ITERATION_LOG_DIR}", makes_new)
    new_dir_maker(f"{LOG_DIR}/{TRAINING_LOG_DIR}", makes_new)
    new_dir_maker(MODEL_DIR, makes_new)
    
    

def get_process_set_dict(
    k_size=K_SIZE, model_dir=MODEL_DIR, iter_log_key=ITER_LOG_KEY,
    train_log_dir=f"{LOG_DIR}/{TRAINING_LOG_DIR}",
    model_save=True, save_train_log=True, save_iter_time_log=True, verbose=True
):
    result = {
        "k_size":k_size,
        
        "model_dir":model_dir,
        "model_save":model_save,
        
        "iter_log_key":iter_log_key,
        "save_iter_time_log":save_iter_time_log,
        
        "train_log_dir":train_log_dir,
        "save_train_log":save_train_log,
        
        "verbose":verbose
    }
    return result

    
    
def get_model_set_dict(
    model_key, cnn_key, optimizer_key, 
    num_class=NUMBER_OF_CLASS,
    use_AMP=USE_AMP,
    epochs=EPOCHS,
    max_norm=GRADIENT_CLIIPING_MAX_NORM,
    earlyStopping=EARLY_STOPPING,
    patience=PATIENCE
):
    result = {
        "model_key":model_key,
        "cnn_key":cnn_key,
        "optimizer":optimizer_key,
        "num_class":num_class,
        "use_AMP":use_AMP,
        "epochs":epochs,
        "max_norm":max_norm,
        "earlyStopping":earlyStopping,
        "patience":patience,
    }
    return result



def get_HP_set_dict(
    learing_rate=0.00001, weight_decay=0.0005,
    T_0=20, T_mult=2, eta_min=0.0000001,
    CS_threshold=0.50, iou_threshold=0.50
):
    """
    2) weight_decay
    >>> 학습 중 가중치 감소를 적용하는 방법.
        주로 Overfitting 방지에 사용되며, 모델의 복잡성을 제한하고 학습 데이터에 대한
        과적합을 줄이는데 도움 됨.
    >>> L2 정규화와 유사하며, 모델 파라미터의 크기를 줄여 일반화 능력 향상 가능
    3) T_0(integer)
    >>> 초기 학습률 감소 Cycle의 길이 정의
        시작 시, 학습률이 감소하는 Epoch(전체 데이터셋을 한 번 학습하는 것)의 수를 결정.
        T_0가 끝나면 학습률이 재설정되며, 다음 Cycle 시작.
    >>> 너무 작으면, 학습률이 너무 빠르게 감소해 충분한 학습이 이뤄지지 않을 수 있음.
        너무 크면, 학습률이 느리게 감소해 학습이 지나치게 오래 걸릴 수 있음.
    >>> 데이터셋이 크고 복잡할수록 더 긴 T_0(예:10-50)가 필요할 수 있음.
        데이터셋이 작고 덜 복잡한 경우 더 짧은 T_0(예:5-10)가 필요할 수 있음.
    4) T_mult(integer)
    >>> 각 재시작 후 Cycle의 길이가 어떻게 증가할지 결정.
        각 Cycle 후 T_0의 배수로, T_mult가 2라면 각 Cycle의 길이는 이전 Cycle의 2배가 됨.
    >>> 너무 크면, 학습률이 너무 오랜 기간 동안 낮은 상태로 유지될 수 있음.
        너무 작으면, 학습률의 변화가 지나치게 빈번해져 모델이 최적해를 찾는데 방해될 수 있음.
    >>> 일반적으로 1 이상의 값을 가지며, 1인 경우 모든 Cycle의 길이가 동일.
        값이 1보다 크면, Cycle의 길이는 각 재시작 후에 저맟 증가
    5) eta_min
    >>> 학습률이 감소하는 동안의 최소값
        너무 높으면 학습률이 충분히 낮아지지 않아 세밀한 최적화가 어려울 수 있음.
        너무 낮으면 학습 과정에서 충분한 탐색이 이뤄지지 않을 수 있음.
    >>> 일반적으로 매우 낮은 값으로 설정되며, 학습률이 너무 낮아져 학습이 정체되는 것을
        방지하면서도, 세밀한 최적화를 위해 충분히 낮아야 함.
    """
    result={
        "learning_rate":learing_rate,
        "weight_decay":weight_decay,
        "T_0":T_0,
        "T_mult":T_mult,
        "eta_min":eta_min,
        "CS_threshold":CS_threshold,
        "iou_threshold":iou_threshold
    }
    return result



