from .imports import * 


class EarlyStopping:
    def __init__(self,
                 monitor_field: str,
                 tolerance_epochs: int,
                 expected_trend: Literal['asc', 'desc']):
        self.tolerance_epochs = tolerance_epochs
        self.expected_trend = expected_trend
        self.monitor_field = monitor_field

        self.best_val = 0. 
        self.best_epoch = -1 
        self.best_dict = {} 
        self.should_stop = False 
        
    def record(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, (FloatArray, FloatTensor)):
                kwargs[key] = float(val)
                
        epoch = kwargs['epoch']
        val = kwargs[self.monitor_field]
            
        if self.best_epoch < 0:
            self.best_epoch = epoch 
            self.best_val = val 
            self.best_dict = kwargs
        else:
            if self.expected_trend == 'asc':
                if val > self.best_val:
                    self.best_val = val 
                    self.best_epoch = epoch 
                    self.best_dict = kwargs
            elif self.expected_trend == 'desc':
                if val < self.best_val:
                    self.best_val = val 
                    self.best_epoch = epoch 
                    self.best_dict = kwargs
            else:
                raise AssertionError

            if epoch - self.best_epoch > self.tolerance_epochs:
                self.should_stop = True 
                
    def check_stop(self, print_result: bool = True) -> dict[str, Any]:
        if self.should_stop:
            if print_result:
                print("End of training, the best result:")
                print(self.best_dict)         

            return self.best_dict
        else:
            return dict() 
        
    def auto_stop(self):
        if self.check_stop(print_result=True):
            exit() 
