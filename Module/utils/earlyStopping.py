import torch
import numpy as np



class EarlyStopping:
    
    def __init__(self, patience:int=5, verbose:bool=False, delta:float=0, ascending:bool=True, filePath:str='checkpoint.pt'):
        """
        Args:
            patience: 성능 개선이 관찰되지 않는 epoch의 수.
            verbose: early stopping 메시지 출력 여부.
            delta: 이전 최고 성능과 비교해서 최소 변화량.
            filePath: checkpoint의 저장 경로.
            ascending: 작은 값일수록 성능이 좋은 것을 의미하는 경우.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.filePath = filePath
        self.ascending = ascending
        
        self.wait_count = 0              # pateince에 근사하게 되는지 count
        self.best_score = None
        self.stop = False
        self.before_best_score = np.Inf
        self.best_epoch = None
        
        
    
    def __call__(self, score, model, epoch=None):
        
        benchmark_score = -score if self.ascending else score
        
        if self.best_score is None:
            self.best_score = benchmark_score
            self.save_checkpoint(model, score)
            self.best_epoch = epoch
            
        elif benchmark_score < self.best_score + self.delta:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stop = True
        else:
            self.save_checkpoint(model, score)
            self.best_score = benchmark_score
            self.wait_count = 0
            self.best_epoch = epoch
            
            
            
    def save_checkpoint(self, model, score):
        if self.verbose:
            print(f"save check point: [before]: {self.before_best_score:.6f} / [now]: {score:.6f} - Saving model...")
        torch.save(model.state_dict(), self.filePath)
        self.before_best_score = score
        
        
        
    def load_checkpoint(self, model):
        print(f"load check point: [epoch]: {self.best_epoch} - Loading model...")
        model.load_state_dict(torch.load(self.filePath))
        
    
    