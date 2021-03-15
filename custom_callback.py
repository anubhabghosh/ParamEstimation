import torch
import numpy as np

def check_patience_fn(best_val_loss_past, best_val_loss_present, threshold):

    abs_change = ((best_val_loss_past - best_val_loss_present) / best_val_loss_past).abs()
    #print("Abs. change after suitable improvement in val.loss :{}".format(rel_change))
    if abs_change > 0 and abs_change < threshold:
        return True
    else:
        return False

        
def callback_val_loss(model, best_model_wts, val_loss, best_val_loss, 
                    current_epoch, patience, num_patience, 
                    min_delta, check_patience=False, orig_stdout=None):
    
    #TODO: Fix this function

    abs_change_in_val_loss = ((best_val_loss - val_loss) / best_val_loss).abs()
    print("Abs. change in val_loss:{}".format(abs_change_in_val_loss))

    if relative_change_in_val_loss > 0 or check_patience == True: # Initially was 2 % decrease in value
                    
        check_patience = True # Set patience checking flag to be True
        
        if patience >= 1 and patience < num_patience:

            abs_change_diff = check_patience_fn(best_val_loss_past=best_val_loss, best_val_loss_present=val_loss, threshold=min_delta)

            if abs_change_diff == True:
                
                patience += 1
                print("Patience:{}".format(patience))
                print("Patience:{}".format(patience), file=orig_stdout)
                if patience == num_patience:
                    
                    print("Saving Model at Epoch {} with val loss: {} ...".format(current_epoch+1, val_loss))
                    print("Saving Model at Epoch {} with val loss: {} ...".format(current_epoch+1, val_loss), file=orig_stdout)
                    
                    #best_val_loss = val_loss 
                    best_val_epoch = current_epoch + 1 
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience = 0
                    check_patience = False
            
            else:
                # Reset patience in case consecutive criterion is not fulfilled
                print("Reset patience")
                print("Reset patience", file=orig_stdout)
                
                patience = 0
                check_patience = False

        elif patience == 0:
            # If the update is happening for the first time
            patience += 1
    
    else:
       # print("Change is not valid")
        check_patience = False

    
    # Update best val loss using current value of validation loss    
    best_val_loss = val_loss

    return best_val_loss, best_model_wts, best_val_epoch, patience, check_patience

