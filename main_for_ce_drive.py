# This code writes down the solution for Task 1
import numpy as np
from torch.utils import data
from utils.data_utils import Series_Dataset, obtain_tr_val_test_idx
from utils.data_utils import get_dataloaders, load_saved_dataset, load_splits_file, NDArrayEncoder
from utils.data_utils import create_splits_file_name
import pickle as pkl 
import os
import torch
import json
from src.rnn_models import RNN_model, train_rnn, evaluate_rnn
import argparse
from parse import parse
import datetime

def check_if_dir_or_file_exists(file_path, file_name=None):
    flag_dir = os.path.exists(file_path)
    if not file_name is None:
        flag_file = os.path.isfile(os.path.join(file_path, file_name))
    else:
        flag_file = None
    return flag_dir, flag_file

def get_date_and_time():
    
    now = datetime.now()
    #print(now)
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python train.py --mode [train/test]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    parser.add_argument("--model_file_saved", help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str)
    parser.add_argument("--data", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--use_norm", help="Use_normalization", type=int, default=None)
    parser.add_argument("--splits_file", help="Enter full path to splits file", type=str, default="splits_file.pkl")
    args = parser.parse_args() 
    mode = args.mode
    model_type = args.model_type
    datafile = args.data
    dataset_type, num_trajectories, num_realizations, N_seq = parse("ce_drive_trajectories_data_{}_M{:d}_P{:d}_N{:d}.pkl", datafile.split('/')[-1])
    datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))
    usenorm_flag = args.use_norm
    model_file_saved = args.model_file_saved
    splits_file = args.splits_file

    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'create_dataset_revised.py' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present")
        Z_pM = load_saved_dataset(filename=datafile)
    
    Z_pM_dataset = Series_Dataset(Z_pM_dict=Z_pM)

    if splits_file is None or not os.path.isfile(splits_file):
        tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_pM_dataset,
                                                                    tr_to_test_split=0.9,
                                                                    tr_to_val_split=0.833)
        print(len(tr_indices), len(val_indices), len(test_indices))
        splits = {}
        splits["train"] = tr_indices
        splits["val"] = val_indices
        splits["test"] = test_indices
        splits_file_name = create_splits_file_name(dataset_filename=datafile,
                                                splits_filename=splits_file
                                                )
        with open(os.path.join(datafolder, splits_file_name), 'wb') as handle:
            pkl.dump(splits, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("Loading the splits file from {}".format(splits_file))
        splits = load_splits_file(splits_filename=splits_file)
        tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]

    batch_size = 128 # Set the batch size
    train_loader, val_loader, test_loader = get_dataloaders(Z_pM_dataset, batch_size, tr_indices, val_indices, test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    #with open("./config/configurations.json") as f: # Config file for estimating all theta parameters
    #    options = json.load(f)

    with open("./config/configurations_alltheta_ce_drive_prbs.json") as f: # Config file for estimating theta_vector when some parameters are fixed
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))
    
    # Define logfile paths and model file paths
    current_date = get_date_and_time()
    dd, mm, yy, hr, mins, secs = parse("{}/{}/{} {}:{}:{}", current_date)
    logfile_path = "./log/ce_drive/{}/{}_L{}_H{}_results_{}{}{}_{}{}{}/".format(dataset_type, 
                                                                    model_type, 
                                                                    options[model_type]["n_layers"], 
                                                                    options[model_type]["n_hidden"],
                                                                    dd, mm, yy, hr, mins, secs)

    modelfile_path = "./models/ce_drive/{}/{}_L{}_H{}_results_{}{}{}_{}{}{}/".format(dataset_type,
                                                                    model_type, 
                                                                    options[model_type]["n_layers"], 
                                                                    options[model_type]["n_hidden"],
                                                                    dd, mm, yy, hr, mins, secs)

    log_file_name = "training_{}_M{}_P{}_N{}.log".format(model_type,
                                                        num_trajectories,
                                                        num_realizations,
                                                        N_seq)

    flag_models_dir, _ = check_if_dir_or_file_exists(modelfile_path,
                                                    file_name=None)
    
    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(logfile_path,
                                                            file_name=log_file_name)
        
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    
    print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))
    
    if flag_log_dir == False:
        print("Creating {}".format(logfile_path))
        os.makedirs(logfile_path, exist_ok=True)
    
    if flag_models_dir == False:
        print("Creating {}".format(modelfile_path))
        os.makedirs(modelfile_path, exist_ok=True)

    if mode.lower() == "train": 
        model_gru = RNN_model(**options[model_type])
        tr_verbose = True  
        tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model = train_rnn(options=options[model_type], 
                                                                                            nepochs=options[model_type]["num_epochs"],
                                                                                            train_loader=train_loader,
                                                                                            val_loader=val_loader,
                                                                                            device=device,
                                                                                            usenorm_flag=usenorm_flag,
                                                                                            tr_verbose=tr_verbose,
                                                                                            save_chkpoints="all",
                                                                                            logfile_path=os.path.join(logfile_path, log_file_name),
                                                                                            modelfile_path=modelfile_path)
        #if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)
        
        losses_model = {}
        losses_model["tr_losses"] = tr_losses
        losses_model["val_losses"] = val_losses

        #with open('./plot_data/{}_losses_eps{}.json'.format(model_type, options[model_type]["num_epochs"]), 'w') as f:
        #    f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":

        #model_file_saved = "./model_checkpoints/{}_usenorm_{}_ckpt_epoch_{}.pt".format(model_type, usenorm_flag, epoch_test)
        evaluate_rnn(options[model_type], test_loader, device, model_file=model_file_saved, usenorm_flag=usenorm_flag)
    
    return None


if __name__ == "__main__":
    main()

    
