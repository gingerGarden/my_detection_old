import json
import numpy as np
import matplotlib.pyplot as plt
from Module.utils.log_utils import my_time_log_progressbar
from Module.utils.Convenience_Function import time_checker


def make_train_log_dict(train_loss_dict, valid_loss_dict, valid_score_dict):

    def add_new_string_to_dictionary_key(new_str, origin_dict):
        stack_dict = dict()
        for key, item in origin_dict.items():
            new_key = f"{new_str}_{key}"
            stack_dict[new_key] = item
        return stack_dict

    new_train_dict = add_new_string_to_dictionary_key(new_str="train", origin_dict=train_loss_dict)
    new_train_dict["train_total"] = sum(loss for loss in new_train_dict.values())
    new_valid_dict = add_new_string_to_dictionary_key(new_str="valid", origin_dict=valid_loss_dict)
    new_valid_dict["valid_total"] = sum(loss for loss in new_valid_dict.values())

    log_dict = new_train_dict | new_valid_dict | valid_score_dict
    return log_dict



class model_iteration_log:
    
    def __init__(self, optimizer, epochs, verbose, save, log_path, log_freq):
        
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.save = save
        self.log_path = log_path
        self.log_freq = log_freq
        
        
        
    def get_log_instances(self, epoch, key="train"):
        
        iter_str_len = len(str(self.epochs))
        header_str = f"Epochs:{epoch+1:>{iter_str_len}}/{self.epochs} [{key}]"
#         header_str = "Epochs: %i/%i [%s]" % (epoch+1, self.epochs, key)
        # Log Instance 생성
        Time_Log_Ins = my_time_log_progressbar(
            header=header_str,
            verbose=self.verbose,
            sep_next=False,
            num_size=4
        )
        # Loss 전처리기 Instance 생성
        Loss_Log_Ins = loss_log()
        return Time_Log_Ins, Loss_Log_Ins
    
    
    
    def one_iteration_log(self, epoch, iteration, time_log_ins, loss_log_ins, loss_dict):

        if self.save:
            self.add_iteration_time_log(epoch, iteration, time_log_ins)
        loss_log_ins.add_loss_dict(loss_dict)   # stack Loss Log
        
        return iteration + 1
    
    
    
    # []로 구성된 json log 파일 생성
    def make_log_file(self):
        if self.save:
            with open(self.log_path, 'w') as file:
                json.dump([], file)
        
        
        
    # 한 iteration에 대한 시간 log 생성
    def add_iteration_time_log(self, epoch, iteration, time_log_ins):
        # log_instance의 time들을 모두 해당 시점으로 update한다.
        time_log_dict = time_log_ins.put_it_at_the_end_of_the_iteration_process()
        # log 생성
        if (iteration%self.log_freq == 0) or (iteration==time_log_ins.iter_size-1):
            log = self.iter_log_dictionary_maker(epoch, iteration, time_log_dict)
            self.overwrite_train_log(log)

            
            
    # log 파일을 불러와서 신규 로그를 추가한다.
    def overwrite_train_log(self, log):

        # json file을 읽는다.
        with open(self.log_path, 'r') as file:
            data = json.load(file)
        # 신규 log 추가
        data.append(log)
        # 변경된 내용 저장
        with open(self.log_path, 'w') as file:
            json.dump(data, file)  # indent=4는 들어쓰기로, json의 가독성을 올려준다.
            
            
            
    # log dictionary를 만든다.
    def iter_log_dictionary_maker(self, epoch, iteration, time_log_dict):

        log = dict()
        log["epoch"] = epoch
        log["iteration"] = iteration
        log["lr"] = "{:1.8f}".format(self.optimizer.param_groups[0]["lr"])
        # 시간 관련 변수 추가
        log["eta"] = time_log_dict['eta']
        log["elapsed"] = time_log_dict['stack']
        log["load"] = time_log_dict['data_load']
        log["iter_train"] = time_log_dict['iter_train']
        return log
        

        
        


class loss_log:
    
    def __init__(self):
        self.stack_dict = None
        self.total_list = []

        
        
    def add_loss_dict(self, loss_dict):
        # stack_dict의 구조가 없는 초기 상태인 경우.
        if self.stack_dict is None:
            self.make_stack_dict_frame(loss_dict)
        # loss_dict을 stack_dict에 쌓는다.
        for key, value in loss_dict.items():
            self.stack_dict[key].append(value.item())
        # loss에 대한 합을 list에 넣는다.
        self.total_list.append(self.make_total_loss(loss_dict))
            
            
            
    def make_stack_dict_frame(self, loss_dict):
        self.stack_dict = dict()
        for key in loss_dict.keys():
            self.stack_dict[key] = []
        
        
        
    def make_total_loss(self, loss_dict):
        total_loss = sum(loss for loss in loss_dict.values())
        return total_loss
        
        
        
    def make_mean_loss_dict(self):
        mean_loss_dict = dict()
        for key, value in self.stack_dict.items():
            mean_loss_dict[key] = np.mean(value)
        
        return mean_loss_dict
    
        
        
    def print_typical_log_sentence(self, start_time, loss_dict, score_dict=None):
        
        total_loss = self.make_total_loss(loss_dict)
        spend_time = time_checker(start_time)
        # 문장 생성의 key
        if score_dict is not None:
            source_dict = score_dict
            key = " [Score] "
        else:
            source_dict = loss_dict
            key = " [Loss ] "

        # 부가 sentence 생성
        sub_sentence = self.source_dict_to_sentence(source_dict)
        log_sentence = " [time] " + spend_time + " total loss: " + "{:1.4f}".format(total_loss) + key + sub_sentence
        print(log_sentence)


        
    def print_test_score_sentence(self, start_time, score_dict):

        spend_time = time_checker(start_time)
        score_sentence = self.source_dict_to_sentence(score_dict)
        log_sentence = " [time] " + spend_time + " [Score] " + score_sentence
        print(log_sentence)
        
        

    def source_dict_to_sentence(self, source_dict):

        sentence = ""
        for key, value in source_dict.items():

            value_str = "{:1.4f}".format(value)
            key_value_str = f"{key} {value_str}"
            sentence = sentence + " | " + key_value_str

        return sentence
        
        
        
    def draw_learning_curve(self, _figsize=(12,8)):

        # total loss 생성
        total_have = "total" in self.stack_dict.keys()
        if total_have != True:
            self.add_total_loss_list()

        plt.figure(figsize=_figsize)
        plt.grid(True) # 교차선
        for key, value in self.stack_dict.items():
            plt.plot(value, label=key)

        plt.title("Train Loss Learning Curve", fontsize=20, pad=20)
        plt.xlabel("Epoch", fontsize = 15, labelpad = 20)
        plt.ylabel("Loss", fontsize = 15, labelpad = 20)
        plt.legend(loc="upper right", fontsize=15)
        plt.show()
        
        
    def add_total_loss_list(self):
        key_list = list(self.stack_dict.keys())
        base_array = np.array(self.stack_dict[key_list[0]])
        for key in key_list[1:]:
            base_array = base_array + np.array(self.stack_dict[key])
        self.stack_dict["total"] = base_array