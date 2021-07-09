# This code writes down the solution for Task 1
import numpy as np
import scipy
from scipy.optimize import minimize
import string
import random
import matplotlib.pyplot as plt
from utils.data_utils import Series_Dataset, obtain_tr_val_test_idx
from utils.data_utils import get_dataloaders, load_saved_dataset, NDArrayEncoder
from utils.plot_utils import plot_trajectories, plot_losses
import pickle as pkl 
import os
import torch
import json
from src.rnn_models import RNN_model, train_rnn, evaluate_rnn
# Add weighted MSE based model
from src.rnn_models_with_weightedloss import train_rnn_with_wtdloss, evaluate_rnn_with_wtdloss
import argparse

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python train.py --mode [train/test]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    parser.add_argument("--model_file_saved", help="Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str)
    parser.add_argument("--data", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--use_norm", help="Use_normalization", type=int, default=None)
    parser.add_argument("--use_weighted", help="Use weighted loss for training", type=int, default=0)
    args = parser.parse_args() 
    mode = args.mode
    model_type = args.model_type
    datafile = args.data
    usenorm_flag = args.use_norm
    model_file_saved = args.model_file_saved
    useweighted_flag = args.use_weighted

    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'create_function.py' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present")
        Z_pM = load_saved_dataset(filename=datafile)
    
    Z_pM_dataset = Series_Dataset(Z_pM_dict=Z_pM)

    tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_pM_dataset,
                                                                tr_to_test_split=0.9,
                                                                tr_to_val_split=0.833)
    print(len(tr_indices), len(val_indices), len(test_indices))

    batch_size = 128 # Set the batch size
    train_loader, val_loader, test_loader = get_dataloaders(Z_pM_dataset, batch_size, tr_indices, val_indices, test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    #for i_batch, sample_batched in enumerate(train_loader):
    #    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    #model_type = "gru"
    #with open("configurations_var.json", 'r') as f:
        #options = json.load(f)
    with open("./config/configurations_var_modified.json", 'r') as f:
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    if mode.lower() == "train": 
        model_gru = RNN_model(**options[model_type])
        tr_verbose = True 
        if useweighted_flag == 0: 
            tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model = train_rnn(options=options[model_type], 
                                                                                                nepochs=options[model_type]["num_epochs"],
                                                                                                train_loader=train_loader,
                                                                                                val_loader=val_loader,
                                                                                                device=device,
                                                                                                usenorm_flag=usenorm_flag,
                                                                                                tr_verbose=tr_verbose)
        elif useweighted_flag == 1:
            print("Using Weighted losses for training, but usual MSE for evaluation...\n")
            tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model = train_rnn_with_wtdloss(options=options[model_type], 
                                                                                                nepochs=options[model_type]["num_epochs"],
                                                                                                train_loader=train_loader,
                                                                                                val_loader=val_loader,
                                                                                                device=device,
                                                                                                usenorm_flag=usenorm_flag,
                                                                                                tr_verbose=tr_verbose)
        #if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)
        
        losses_model = {}
        losses_model["tr_losses"] = tr_losses
        losses_model["val_losses"] = val_losses

        with open('./plot_data/{}_losses_eps{}_var.json'.format(model_type, options[model_type]["num_epochs"]), 'w') as f:
            f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":

        #model_file_saved = "./models/{}_usenorm_{}_ckpt_epoch_{}.pt".format(model_type, usenorm_flag, epoch_test)
        if useweighted_flag == 0:
            evaluate_rnn(options[model_type], test_loader, device, model_file=model_file_saved, usenorm_flag=usenorm_flag)
        elif useweighted_flag == 1:
            evaluate_rnn_with_wtdloss(options[model_type], test_loader, device, model_file=model_file_saved, usenorm_flag=usenorm_flag)

    return None


if __name__ == "__main__":
    main()

    
