from Module.Global_variable import time, torch
from Module.utils.Convenience_Function import time_checker
from Module.utils.Convenience_Function_by_torch import image_list_upload_to_device, target_list_upload_to_device



class training_process:
    
    def __init__(self, model, optimizer, scheduler, scaler, dataLoader, log_ins, device, use_amp_bool, max_norm, verbose):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.dataLoader = dataLoader
        self.log_ins = log_ins
        self.device = device
        self.use_amp_bool = use_amp_bool
        self.max_norm = max_norm
        self.verbose = verbose
    
    
    def train_one_epoch(self, epoch):

        Time_Log_Ins, Loss_Log_Ins = self.log_ins.get_log_instances(epoch)  # Get Log instances
        start_time = time.time()
        iteration = 0

        self.model.train()   # model train
        for imgs, targets in Time_Log_Ins.with_time_log(self.dataLoader):

            # 1. upload to device
            imgs = image_list_upload_to_device(img_list=imgs, device=self.device)
            targets = target_list_upload_to_device(target_list=targets, device=self.device)

            # 2. model training - AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp_bool):
                loss_dict = self.model(imgs, targets)
                # sum all loss_dict's loss(loss_classification, loss_box_reg, loss_objectness, loss_rpn_box_reg)
                losses = sum(loss for loss in loss_dict.values())

            # 3. back propagation
            self.back_propagation(losses)
            # 4. Log
            iteration = self.log_ins.one_iteration_log(epoch, iteration, Time_Log_Ins, Loss_Log_Ins, loss_dict)

        # Loss 평균 산출
        mean_loss_dict = Loss_Log_Ins.make_mean_loss_dict()
        if self.verbose:
            Loss_Log_Ins.print_typical_log_sentence(start_time, loss_dict=mean_loss_dict)

        return mean_loss_dict
                
        
        
    def back_propagation(self, losses):
        self.optimizer.zero_grad()
        # 3.1. AMP - GradScale use or not
        if self.use_amp_bool:
            self.gradient_scaled_parameter_update_with_clipping(losses=losses)
        else:
            self.parameter_update_with_clipping(losses=losses)
        # 3.2. step scheduler - CosinAnnelingWarmRestarts(Iterantion scheduler)
        self.scheduler.step()
        
        
        
    def gradient_scaled_parameter_update_with_clipping(self, losses):
        # Gradient scaling with back propagation
        self.scaler.scale(losses).backward()
        # Gradient update 전에 Gradient clipping 적용
        #################################################################
        # AMP 사용 시, Gradient clipping은 scaling 역산 후 적용되어야 한다.
        self.scaler.unscale_(self.optimizer) # Scaling 역산
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_norm
        )
        #################################################################
        # The parameters are updated using a scaled gradient
        self.scaler.step(self.optimizer)
        # scaler update
        self.scaler.update()
        
        
        
    def parameter_update_with_clipping(self, losses):
        losses.backward()
        # Gradient update 전 Gradient clipping 적용
        #################################################################
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_norm
        )
        ################################################################
        # parameter update
        self.optimizer.step()