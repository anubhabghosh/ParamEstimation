# This file defines the model to be used for use in parameter estimation
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
from torch.autograd import Variable
from timeit import default_timer as timer
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
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.output_size)
    
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
        
        # Passing the output to the fully connected layer
        y = self.fc(r_out_last_step)
        return y

def save_model(model, filepath):
    
    torch.save(model.state_dict(), filepath)
    return None

def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets
        
def train_rnn(options, nepochs, train_loader, val_loader, device, usenorm_flag=0, tr_verbose=True):
    """ This function implements the training algorithm for the RNN model
    """
    model = RNN_model(**options)
    model = push_model(nets=model, device=device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//2, gamma=0.9)
    criterion = nn.MSELoss() # By default reduction is 'mean'
    tr_losses = []
    val_losses = []
    model_filepath = "./models/gru_L2_H50_results/"
    training_logfile = "./log/training_{}_usenorm_{}.log".format(usenorm_flag, model.model_type)
    best_val_loss = np.inf
    best_model_wts = None

    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp

    #best_model_wts = None
    #best_val_epoch = None
    #check_patience = False
    #num_patience = 4
    #best_val_loss = np.inf
    #patience = 0
    #relative_loss = 0.0
    #threshold = 0.02
    print("------------------------------ Training begins ---------------------------------")
    print("Model config / options:\n{}\n".format(options))
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
            tr_loss_batch = criterion(tr_predictions_batch, tr_targets_batch.squeeze(2).to(device))
            tr_loss_batch.backward()
            optimizer.step()
            #scheduler.step()

            # print statistics
            tr_running_loss += tr_loss_batch.item()
            tr_loss_epoch_sum += tr_loss_batch.item()

            if i % 50 == 49 and ((epoch + 1) % 50 == 0):    # print every 10 mini-batches
                print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 50))
                print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 50), file=orig_stdout)
                tr_running_loss = 0.0
        
        scheduler.step()
        with torch.no_grad():
            
            for i, data in enumerate(val_loader, 0):
                
                val_inputs_batch, val_targets_batch = data
                X_val = Variable(val_inputs_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                val_predictions_batch = model.forward(X_val)
                val_loss_batch = criterion(val_predictions_batch, val_targets_batch.squeeze(2).to(device))
                # print statistics
                val_loss_epoch_sum += val_loss_batch.item()
        
        # Loss at the end of each epoch
        tr_loss = tr_loss_epoch_sum / len(train_loader)
        val_loss = val_loss_epoch_sum / len(val_loader)

        endtime = timer()
        # Measure wallclock time
        time_elapsed = endtime - starttime

        # Displaying loss at an interval of 50 epochs
        if tr_verbose == True and (((epoch + 1) % 100) == 0 or epoch == 0):
            
            print("Epoch: {}/{}, Training MSE Loss:{:.9f}, Val. MSE Loss:{:.9f} ".format(epoch+1, 
            model.num_epochs, tr_loss, val_loss), file=orig_stdout)
            #save_model(model, model_filepath + "/" + "{}_ckpt_epoch_{}.pt".format(model.model_type, epoch+1))

            print("Epoch: {}/{}, Training MSE Loss:{:.9f}, Val. MSE Loss:{:.9f}, Time_Elapsed:{:.4f} secs".format(epoch+1, 
            model.num_epochs, tr_loss, val_loss, time_elapsed))
        
        # Checkpointing the model every 200 epochs
        if (((epoch + 1) % 500) == 0 or epoch == 0):     
            # Checkpointing model every 100 epochs
            save_model(model, model_filepath + "/" + "{}_usenorm_{}_ckpt_epoch_{}.pt".format(model.model_type, usenorm_flag, epoch+1))
        
        # Save best model in case validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch+1
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # Saving losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
    
    # Save the best model as per validation loss at the end
    print("Saving the best model at epoch={}".format(best_epoch))
    torch.save(best_model_wts, model_filepath + "/" + "{}_usenorm_{}_ckpt_epoch_{}_best.pt".format(model.model_type, usenorm_flag, best_epoch))
    sys.stdout = orig_stdout
    return tr_losses, val_losses, model

def evaluate_rnn(options, test_loader, device, model_file=None, usenorm_flag=0):

    te_running_loss = 0.0
    test_loss_epoch_sum = 0.0
    print("################ Evaluation Begins ################ \n")    
    
    # Set model in evaluation mode
    model = RNN_model(**options)
    model.load_state_dict(torch.load(model_file))
    criterion = nn.MSELoss()
    model = push_model(nets=model, device=device)
    model.eval()
    test_log = "./log/test_{}_usenorm_{}.log".format(usenorm_flag, options["model_type"])

    with torch.no_grad():
        
        for i, data in enumerate(test_loader, 0):
                
            te_inputs_batch, te_targets_batch = data
            X_test = Variable(te_inputs_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            test_predictions_batch = model.forward(X_test)
            test_loss_batch = criterion(test_predictions_batch, te_targets_batch.squeeze(2).to(device))
            # print statistics
            test_loss_epoch_sum += test_loss_batch.item()

    test_loss = test_loss_epoch_sum / len(test_loader)

    print('Test loss: {:.3f} using weights from file: {} %'.format(test_loss, model_file))

    with open(test_log, "w") as logfile_test:
        logfile_test.write('Test loss: {:.3f} using weights from file: {}'.format(test_loss, model_file))
