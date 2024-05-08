from Module.Global_variable import torch, np, pd

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Module.process1.model_training import training_process
from Module.process1.model_log import model_iteration_log, loss_log, make_train_log_dict
from Module.process1.model_evaluate import model_evaluate
from Module.process1.torch_dataset import get_my_dataLoader
from Module.process1.torch_basic_style_model import get_my_torch_model, get_optimizer

from Module.utils.Convenience_Function_by_torch import torch_device, state_dict_to_np_array_dict
from Module.utils.Convenience_Function import get_RGB_image_by_cv2, draw_img_and_bbox_torch_style
from Module.utils.earlyStopping import EarlyStopping
from Module.utils.log_utils import make_new_json_list_log_file, append_log_dict_to_json_list_log_file




class model_pipeline:
    
    def __init__(self, p_set_dict, m_set_dict, hp_dict, idx_dict, gpu_num, use_compile=False, log_freq=5):
        
        self.p_set_dict = p_set_dict
        self.m_set_dict = m_set_dict
        self.hp_dict = hp_dict
        self.idx_dict = idx_dict
        self.loader = get_my_dataLoader(idx_dict)
        self.gpu_num = gpu_num
        self.device = torch_device().get_device(gpu_number=gpu_num)
        self.use_compile = use_compile
        self.log_freq = log_freq
        self.check_point = f"checkpoint_{gpu_num}.pt"
        
        self.pred_label_idx_dict = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.Train_Ins = None
        self.Iter_log_Ins = None
        self.early_stopping = None
        self.Eval_Ins = None
        self.train_log_path = None
        self.model_path = None
        
        
        
    def model_train_and_evaluate(self):

        train_loss = loss_log()
        valid_loss = loss_log()

        for epoch in range(self.m_set_dict["epochs"]):
            # Model training
            train_loss_dict = self.Train_Ins.train_one_epoch(epoch)
            train_loss.add_loss_dict(train_loss_dict)

            # Model validation
            valid_loss_dict, valid_score_dict = self.Eval_Ins.valid_one_epoch(epoch)
            valid_loss.add_loss_dict(valid_loss_dict)

            # Early Stopping
            if self.early_stopping is not None:
                self.early_stopping(score=valid_loss.total_list[epoch], model=self.model, epoch=epoch)
                if self.early_stopping.stop:
                    break
            # Log 저장
            self.make_train_log_and_save(train_loss_dict, valid_loss_dict, valid_score_dict)

        # 가장 좋은 모델을 불러온다.
        if self.early_stopping is not None:
            self.early_stopping.load_checkpoint(self.model)

        # 최종 평가.
        _, valid_score_dict = self.Eval_Ins.valid_one_epoch(epoch)
        predict_dict, test_score_dict = self.Eval_Ins.model_test(epoch)

        # 결과 정리
        loss_dict, score_dict = self.model_training_result_handler(train_loss, valid_loss, valid_score_dict, test_score_dict)
        if self.p_set_dict['model_save']:
            torch.save(self.model.state_dict(), self.model_path)

        return loss_dict, score_dict, predict_dict
        
        
        
    # k번째에 해당하는 Loader와 idx_dict를 가지고 온다.
    def define_k_Loader_and_idx_dict(self, k):
        # k에 해당하는 DataLoader 정의
        self.train_loader, self.valid_loader, self.test_loader\
        = self.loader.get_all_torch_dataLoader(k)
        
        # k에 해당하는 predict 대상 idx_dict 정의
        k_idx_dict = self.idx_dict[f"{self.loader.data_key}{k}"]
        self.pred_label_idx_dict = self.get_predict_label_idx_dict(k_idx_dict)
        
        
        
    def get_predict_label_idx_dict(self, k_idx_dict):
        result = dict()
        # valid와 test의 index_dict을 하나로 합친다.
        ## image에 대하여 image_id를 key로 하여 합친다.
        images_list = k_idx_dict['valid']['images'] + k_idx_dict['test']['images']
        result["id_image"] = pd.DataFrame(images_list).set_index("id").to_dict(orient="index")
        ## annotation
        result["annotation"] = k_idx_dict['valid']['annotations'] | k_idx_dict['test']['annotations']
        ## image의 부모 경로
        result["img_root"] = self.loader.img_root
        return result
        
        
        
    def import_model(self):
        # Model 정의
        model = get_my_torch_model(
            class_num=self.m_set_dict['num_class'],
            model_key=self.m_set_dict['model_key'],
            cnn_key=self.m_set_dict['cnn_key']
        ).process().to(self.device)
        # torch 2.0의 compile 사용 여부 - Object detection model에서 속도가 느려지는 이슈가 있었음.
        self.model = torch.compile(model) if self.use_compile else model
        
        
        
    def import_optimizer_and_scheduler(self):
        
        # Optimizer 정의
        self.optimizer = get_optimizer(
            self.model,
            learning_rate = self.hp_dict['learning_rate'],
            weight_decay=self.hp_dict['weight_decay'],
            opt_key=self.m_set_dict['optimizer']
        )
        # Scheduler 정의
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=self.hp_dict['T_0'],
            T_mult=self.hp_dict['T_mult'],
            eta_min=self.hp_dict['eta_min']
        )
        # scaler(AMP) 정의 - Autocast 후 Gradient scaling 적용.
        self.scaler = torch.cuda.amp.GradScaler()
        
        
        
    def make_important_Instance(self, k):

        self.Iter_log_Ins = model_iteration_log(
            optimizer=self.optimizer,
            epochs=self.m_set_dict['epochs'],
            verbose=self.p_set_dict['verbose'],
            save=self.p_set_dict['save_iter_time_log'],
            log_path=self.p_set_dict['iter_log_key'] + "_" + str(self.gpu_num) + f"_{k}.json",
            log_freq=self.log_freq
        )
        self.Train_Ins = training_process(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            dataLoader=self.train_loader,
            log_ins=self.Iter_log_Ins,
            device=self.device,
            use_amp_bool=self.m_set_dict['use_AMP'],
            max_norm=self.m_set_dict['max_norm'],
            verbose=self.p_set_dict['verbose']
        )
        # 모델의 평가를 위한 Instance 생성    
        self.Eval_Ins = model_evaluate(
            model=self.model,
            valid_Loader=self.valid_loader,
            test_Loader= self.test_loader,
            device=self.device,
            CS_threshold=self.hp_dict['CS_threshold'],
            iou_threshold=self.hp_dict['iou_threshold'],
            label_idx_dict=self.pred_label_idx_dict,
            log_ins=self.Iter_log_Ins,
            verbose=self.p_set_dict['verbose']
        )
        # early stopping instance 생성
        if self.m_set_dict["earlyStopping"]:
            self.early_stopping = EarlyStopping(
                patience = self.m_set_dict['patience'],
                verbose = self.p_set_dict['verbose'],
                filePath = self.check_point
            )
        else:
            self.early_stopping = None
            
        
        
    # train log가 저장될 경로와 파일을 생성한다.
    def make_train_log_path_and_file(self, k):
        
        if self.p_set_dict['save_train_log']:
            
            log_dir = self.p_set_dict['train_log_dir']
            model_key = self.m_set_dict["model_key"]
            cnn_key = self.m_set_dict["cnn_key"]
            self.train_log_path = f"{log_dir}/{model_key}_{cnn_key}_{self.gpu_num}_{k}.json"
            # log file 생성
            make_new_json_list_log_file(log_path=self.train_log_path)
            
            
            
    # model 저장 경로 생성
    def make_model_save_path(self, k):
        
        if self.p_set_dict['model_save']:
            
            model_dir = self.p_set_dict['model_dir']
            model_key = self.m_set_dict["model_key"]
            cnn_key = self.m_set_dict["cnn_key"]
            self.model_path = f"{model_dir}/{model_key}_{cnn_key}_{self.gpu_num}_{k}.pt"
            
            
            
    # log를 생성하고 저장한다.
    def make_train_log_and_save(self, train_loss_dict, valid_loss_dict, valid_score_dict):
        
        if self.p_set_dict['save_train_log']:
        
            train_log_dict = make_train_log_dict(
                train_loss_dict, valid_loss_dict, valid_score_dict)
            append_log_dict_to_json_list_log_file(
                log_path=self.train_log_path, log_dict=train_log_dict)
            
            
            
    def model_training_result_handler(
        self, train_loss, valid_loss, valid_score_dict, test_score_dict
    ):
        loss_dict = {
            "total":{
                "train":train_loss.total_list,
                "valid":valid_loss.total_list
            },
            "detail":{
                "train":train_loss.stack_dict,
                "valid":valid_loss.stack_dict
            }
        }
        score_dict = {
            "valid":valid_score_dict,
            "test":test_score_dict
        }
        return loss_dict, score_dict