import numpy as np
from collections import deque

class ConvergenceMonitor_ES(object):

    def __init__(self, tol=1e-2, max_epochs=3):

        self.tol = tol
        self.max_epochs = max_epochs
        self.convergence_flag = False
        self.epoch_arr = [] # Empty list to store iteration numbers to check for consecutive iterations
        self.epoch_count = 0 # Counts the number of consecutive iterations
        self.epoch_prev = 0 # Stores the value of the previous iteration index
        self.history = deque()

    def record(self, current_loss):

        if np.isnan(current_loss) == False:
            
            # In case current_loss is not a NaN, it will continue to monitor
            if len(self.history) < 2:
                self.history.append(current_loss)
            elif len(self.history) == 2:
                _ = self.history.popleft()
                self.history.append(current_loss)
        
        else:
            # Empty the queue in case a NaN loss is encountered during training
            for _ in range(len(self.history)):
                _ = self.history.pop()
    
    #def check_convergence(self):
    def check_worsening_val_error(self):

        #if (abs(self.history[0]) > 0) and (abs((self.history[0] - self.history[-1]) / self.history[0]) < self.tol):
        if (abs(self.history[0]) > 0) and (abs(self.history[-1]) > 0) and ((self.history[-1] - self.history[0]) > self.tol):
            worsening_flag = True
        else:
            worsening_flag = False
        return worsening_flag

    def monitor(self, epoch):

        if len(self.history) == 2 and self.convergence_flag == False:
            
            worsening_flag = self.check_worsening_val_error()                                                                     
            if worsening_flag == True and self.epoch_prev == epoch-1: # If convergence is satisfied
                self.epoch_count += 1 
                self.epoch_arr.append(epoch)
                if self.epoch_count == self.max_epochs: # Patience has exceeded!!
                    print("Consecutive iterations are:{}".format(self.epoch_arr))
                    print("Exit and Convergence reached after {} iterations for relative change in NLL below :{}".format(self.epoch_count, self.tol))  
                    self.worsening_flag = True 
                
            else:
                #print("Consecutive criteria failed, Buffer Reset !!")
                #print("Buffer State:{} reset!!".format(self.epoch_arr)) # Display the buffer state till that time
                self.epoch_count = 0 # Resets the epoch counter
                self.epoch_arr = [] # Resets the epoch counter array
                self.worsening_flag = False # sets the flag to be false
                self.history = deque() # Resets the history deque

            self.epoch_prev = epoch # Set iter_prev as the previous iteration index
        
        else:
            pass
        
        return self.worsening_flag
