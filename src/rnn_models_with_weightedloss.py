# This file defines the model to be used for use in parameter estimation
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
from torch.autograd import Variable
from timeit import default_timer as timer
from data_utils import sample_parameter_modified
import copy
#from tqdm import tqdm

# Create an RNN model for prediction
class RNN_model(nn.Module):
    """ This super class defines the specific model to be used i.e. LSTM or GRU or RNN
    """
    def __init__(self, input_size, output_size, n_hidden, n_layers, 
        model_type, lr, num_epochs, num_directions=1, batch_first = True):
        super(RNN_model, self).__init__()
        """
        Args:
        - input_size: The dimensionality of the input data
        - output_size: The dimensionality of the output data
        - n_hidden: The size of the hidden layer, i.e. the number of hidden units used
        - n_layers: The number of hidden layers
        - model_type: The type of modle used ("lstm"/"gru"/"rnn")
        - lr: Learning rate used for training
        - num_epochs: The number of epochs used for training
        - num_directions: Parameter for bi-directional RNNs (usually set to 1 in this case, for bidirectional set as 2)
        - batch_first: Option to have batches with the batch dimension as the starting dimension 
        of the input data
        """
        # Defining some parameters
        self.hidden_dim = n_hidden  
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        
        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        
        # Predefined:
        ## Use only the forward direction 
        self.num_directions = 1
        
        ## The input tensors must have shape (batch_size,...)
        self.batch_first = True
        
        # Defining the recurrent layers 
        if model_type.lower() == "rnn": # RNN 
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)   
        elif model_type.lower() == "lstm": # LSTM
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "gru": # GRU
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)  
        else:
            print("Model type unknown:", model_type.lower()) 
            sys.exit() 
        
        # Fully connected layer to be used for mapping the output
        #self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.output_size)
        
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, 32)
        self.fc2 = nn.Linear(32, self.output_size)
        # Add a dropout layer with 20% probability
        #self.d1 = nn.Dropout(p=0.2)

    def init_h0(self, batch_size):
        """ This function defines the initial hidden state of the RNN
        """
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h0
    
    def forward(self, x):
        """ This function defines the forward function to be used for the RNN model
        """
        batch_size = x.shape[0]
        
        # Obtain the RNN output
        r_out, hn_all = self.rnn(x)
        
        # Reshaping the output appropriately
        r_out = r_out.contiguous().view(batch_size, -1, self.num_directions, self.hidden_dim)
        
        # Select the last time-step
        r_out = r_out[:, -1, :, :]
        r_out_last_step = r_out.reshape((-1, self.hidden_dim))
        
        # Pass this through dropout layer
        #r_out_last_step = self.d1(r_out_last_step)

        # Passing the output to the fully connected layer
        y = F.relu(self.fc(r_out_last_step))
        y = self.fc2(y)
        #y = self.fc(r_out_last_step)
        return y

def save_model(model, filepath):
    
    torch.save(model.state_dict(), filepath)
    return None

def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets

def weighted_mse_loss(input, target, weight):
    #print("Modified loss shape, before mean ", ((input - target) ** 2).shape)
    #print("Weight used: ", weight)
    return (weight * (input - target) ** 2).mean()

def count_params(model):
    """
    Counts two types of parameters:
    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)
    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    return total_num_params, total_num_trainable_params

def train_rnn_with_wtdloss(options, nepochs, train_loader, val_loader, device, usenorm_flag=0, tr_verbose=True, save_chkpoints=True):
    """ This function implements the training algorithm for the RNN model
    """
    model = RNN_model(**options)
    model = push_model(nets=model, device=device)
    total_num_params, total_num_trainable_params = count_params(model)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//3, gamma=0.9)
    criterion_default = nn.MSELoss() # By default reduction is 'mean'

    criterion = weighted_mse_loss
    weights = torch.from_numpy(sample_parameter_modified()[-1][-2:]) # add weights to device (these weights will weight the MSE loss)
    #weights = weights / (weights.sum() + 1e-12)
    weights = Variable(weights, requires_grad=False).type(torch.FloatTensor).to(device)

    tr_losses = []
    val_losses = []
    model_filepath = "./models/"
    if save_chkpoints == True:
        # No grid search
        training_logfile = "./log/training_{}_usenorm_{}_weightedMSE_var_NS25000.log".format(model.model_type, usenorm_flag)
    else:
        # Grid search
        training_logfile = "./log/gs_training_{}_usenorm_{}_weightedMSE_var_NS25000.log".format(model.model_type, usenorm_flag)
    best_val_loss = np.inf
    tr_loss_for_best_val_loss = np.inf
    best_model_wts = None

    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp

    print("------------------------------ Training begins --------------------------------- \n")
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))
    
    # Start time
    starttime = timer()
    for epoch in range(nepochs):
        
        tr_running_loss = 0.0
        tr_loss_epoch_sum = 0.0
        val_loss_epoch_sum = 0.0
    
        for i, data in enumerate(train_loader, 0):
        
            tr_inputs_batch, tr_targets_batch = data
            optimizer.zero_grad()
            X_train = Variable(tr_inputs_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            tr_predictions_batch = model.forward(X_train)
            tr_loss_batch = criterion(tr_predictions_batch, tr_targets_batch.squeeze(2).to(device), weight=weights)
            #print(weights.shape, tr_predictions_batch.shape, file=orig_stdout)
            tr_loss_batch.backward()
            optimizer.step()
            #scheduler.step()

            # print statistics
            tr_running_loss += tr_loss_batch.item()
            tr_loss_epoch_sum += tr_loss_batch.item()

            if i % 50 == 49 and ((epoch + 1) % 50 == 0):    # print every 10 mini-batches
                #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 50))
                #print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 50), file=orig_stdout)
                tr_running_loss = 0.0
        
        scheduler.step()
        with torch.no_grad():
            
            for i, data in enumerate(val_loader, 0):
                
                val_inputs_batch, val_targets_batch = data
                X_val = Variable(val_inputs_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                val_predictions_batch = model.forward(X_val)
                val_loss_batch = criterion_default(val_predictions_batch, val_targets_batch.squeeze(2).to(device))
                #val_loss_batch = criterion(val_predictions_batch, val_targets_batch.squeeze(2).to(device), weight=weights)
                # print statistics
                val_loss_epoch_sum += val_loss_batch.item()
        
        # Loss at the end of each epoch
        tr_loss = tr_loss_epoch_sum / len(train_loader)
        val_loss = val_loss_epoch_sum / len(val_loader)

        endtime = timer()
        # Measure wallclock time
        time_elapsed = endtime - starttime

        # Displaying loss at an interval of 50 epochs
        if tr_verbose == True and (((epoch + 1) % 200) == 0 or epoch == 0):
            
            print("Epoch: {}/{}, Training MSE Loss:{:.9f}, Val. MSE Loss:{:.9f} ".format(epoch+1, 
            model.num_epochs, tr_loss, val_loss), file=orig_stdout)
            #save_model(model, model_filepath + "/" + "{}_ckpt_epoch_{}.pt".format(model.model_type, epoch+1))

            print("Epoch: {}/{}, Training MSE Loss:{:.9f}, Val. MSE Loss:{:.9f}, Time_Elapsed:{:.4f} secs".format(epoch+1, 
            model.num_epochs, tr_loss, val_loss, time_elapsed))
        
        # Checkpointing the model every 200 epochs
        #if (((epoch + 1) % 500) == 0 or epoch == 0):   
        if (((epoch + 1) % 500) == 0 or epoch == 0) and save_chkpoints == True:     
            # Checkpointing model every 100 epochs, in case of grid_search is being done, save_chkpoints = False
            save_model(model, model_filepath + "/" + "{}_usenorm_{}_ckpt_weightedMSE_epoch_{}.pt".format(model.model_type, usenorm_flag, epoch+1))
        
        # Save best model in case validation loss improves
        if val_loss < best_val_loss:

            best_val_loss = val_loss # Save best validation loss
            tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
            best_epoch = epoch+1 # Corresponding value of epoch
            best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model
        
        # Saving losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
    
    # Save the best model as per validation loss at the end
    print("Saving the best model at epoch={}, with training loss={}, validation loss={}".format(best_epoch, tr_loss_for_best_val_loss, best_val_loss))
    
    if save_chkpoints == True:
        model_filename = "{}_usenorm_{}_ckpt_weightedMSE_epoch_{}_best.pt".format(model.model_type, usenorm_flag, best_epoch)
        # Save the best model using the designated filename
        torch.save(best_model_wts, model_filepath + "/" + model_filename)
    elif save_chkpoints == False:
        pass
    
    print("------------------------------ Training ends --------------------------------- \n")
    # Restoring the original std out pointer
    sys.stdout = orig_stdout

    return tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model

def evaluate_rnn_with_wtdloss(options, test_loader, device, model_file=None, usenorm_flag=0):

    te_running_loss = 0.0
    test_loss_epoch_sum = 0.0
    print("################ Evaluation Begins ################ \n")    
    
    # Set model in evaluation mode
    model = RNN_model(**options)
    model.load_state_dict(torch.load(model_file))
    criterion = nn.MSELoss()
    #criterion = weighted_mse_loss
    #weights = torch.from_numpy(sample_parameter_modified()[-1][-2:])
    #weights = Variable(weights, requires_grad=False).type(torch.FloatTensor).to(device)
    model = push_model(nets=model, device=device)
    model.eval()
    test_log = "./log/test_{}_usenorm_{}_weightedMSE_var.log".format(usenorm_flag, options["model_type"])

    with torch.no_grad():
        
        for i, data in enumerate(test_loader, 0):
                
            te_inputs_batch, te_targets_batch = data
            X_test = Variable(te_inputs_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            test_predictions_batch = model.forward(X_test)
            test_loss_batch = criterion(test_predictions_batch, te_targets_batch.squeeze(2).to(device))
            #test_loss_batch = criterion(test_predictions_batch, te_targets_batch.squeeze(2).to(device), weight=weights)
            # print statistics
            test_loss_epoch_sum += test_loss_batch.item()

    test_loss = test_loss_epoch_sum / len(test_loader)

    print('Test loss: {:.3f} using weights from file: {} %'.format(test_loss, model_file))

    with open(test_log, "w") as logfile_test:
        logfile_test.write('Test loss: {:.3f} using weights from file: {}'.format(test_loss, model_file))
