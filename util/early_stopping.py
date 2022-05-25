from util import * 


class EarlyStopping:
    def __init__(self,
                 monitor_epochs: int = 50):
        self.loss_list: list[float] = []  
        
        self.monitor_epochs = monitor_epochs 

        self.should_stop = False 
        
    def record_loss(self, loss):
        if not isinstance(loss, float):
            loss = float(loss)
            
        self.loss_list.append(loss)
        
        # 如果取最后n次loss，loss一直没下降，则early stop
        if len(self.loss_list) > self.monitor_epochs:
            monitor_list = self.loss_list[-self.monitor_epochs:]
            
            if min(monitor_list) == monitor_list[0]:
                self.should_stop = True 

    def check_stop(self):
        if self.should_stop:
            print("Early Stop!!!")
            exit()
