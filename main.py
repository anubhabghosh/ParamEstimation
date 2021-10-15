# This code calls the functions relevant for running RNN-based parameter estimator on
# NLSS models
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import Series_Dataset, obtain_tr_val_test_idx
from utils.data_utils import get_dataloaders, load_saved_dataset, load_splits_file, NDArrayEncoder
from utils.data_utils import create_splits_file_name, check_if_dir_or_file_exists
import pickle as pkl 
import os
import torch
import json
from parse import parse
#from src.rnn_models import RNN_model, train_rnn, evaluate_rnn
from src.rnn_models_NLSS_modified import RNN_model, train_rnn, evaluate_rnn
import argparse

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python main.py --mode [train/test] --model_type [gru/lstm] --dataset_mode [pfixed/vars/all] \n"\
        "--datafile [fullpath to datafile] --use_norm [0/1] --splits [fullpath to splits file]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    parser.add_argument("--dataset_mode", help="Enter the type of dataset (pfixed/vars/all)", type=str)
    parser.add_argument("--model_file_saved", help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str)
    parser.add_argument("--datafile", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--use_norm", help="Use_normalization", type=int, default=None)
    parser.add_argument("--splits", help="Enter full path to splits file", type=str)
    
    args = parser.parse_args() 
    mode = args.mode
    model_type = args.model_type
    datafile = args.datafile
    dataset_mode = args.dataset_mode
    datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))
    usenorm_flag = args.use_norm
    model_file_saved = args.model_file_saved
    splits_file = args.splits

    # Dataset parameters obtained from the 'datafile' variable
    _, num_trajs, num_realizations, N_seq = parse("{}_M{:d}_P{:d}_N{:d}.pkl", datafile.split('/')[-1])
    batch_size = 128 # Set the batch size
    
    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'create_dataset_revised.py' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present")
        Z_pM = load_saved_dataset(filename=datafile)
    
    Z_pM_dataset = Series_Dataset(Z_pM_dict=Z_pM)

    if not os.path.isfile(splits_file):
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
        with open(splits_file_name, 'wb') as handle:
            pkl.dump(splits, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("Loading the splits file from {}".format(splits_file))
        splits = load_splits_file(splits_filename=splits_file)
        tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]

    train_loader, val_loader, test_loader = get_dataloaders(Z_pM_dataset, batch_size, tr_indices, val_indices, test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    #with open("./config/configurations.json") as f: # Config file for estimating all theta parameters
    #    options = json.load(f)

    with open("./config/configurations_alltheta_pfixed.json") as f: # Config file for estimating theta_vector when some parameters are fixed
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))
    
    logfile_path = "./log/estimate_theta_{}/".format(dataset_mode)
    modelfile_path = "./models/"

    #NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_L{}_H{}_modified_RNN".format(model_type, options[model_type]["n_layers"], options[model_type]["n_hidden"])

    #print(params)
    tr_log_file_name = "training_{}_M{}_P{}_N{}.log".format(model_type,
                                                            num_trajs,
                                                            num_realizations,
                                                            N_seq)
    
    te_log_file_name = "testing_{}_M{}_P{}_N{}.log".format(model_type,
                                                            num_trajs,
                                                            num_realizations,
                                                            N_seq)
    
    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(os.path.join(logfile_path, main_exp_name),
                                                            file_name=tr_log_file_name)
    
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    
    flag_models_dir, _ = check_if_dir_or_file_exists(os.path.join(modelfile_path, main_exp_name),
                                                    file_name=None)
    
    print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))
    
    tr_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), tr_log_file_name)
    te_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), te_log_file_name)

    if flag_log_dir == False:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)
    
    if flag_models_dir == False:
        print("Creating {}".format(os.path.join(modelfile_path, main_exp_name)))
        os.makedirs(os.path.join(modelfile_path, main_exp_name), exist_ok=True)

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
                                                                                            logfile_path=tr_logfile_name_with_path,
                                                                                            modelfile_path=os.path.join(modelfile_path, main_exp_name),
                                                                                            save_chkpoints="some")
        #if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)
        
        losses_model = {}
        losses_model["tr_losses"] = tr_losses
        losses_model["val_losses"] = val_losses

        with open(os.path.join(os.path.join(logfile_path, main_exp_name), 
            '{}_losses_eps{}.json'.format(model_type, options[model_type]["num_epochs"])), 'w') as f:
            f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":

        #model_file_saved = "./model_checkpoints/{}_usenorm_{}_ckpt_epoch_{}.pt".format(model_type, usenorm_flag, epoch_test)
        evaluate_rnn(options[model_type], test_loader, device, model_file=model_file_saved, usenorm_flag=usenorm_flag, 
                    test_logfile_path=te_logfile_name_with_path)
    
    return None

if __name__ == "__main__":
    main()

    
