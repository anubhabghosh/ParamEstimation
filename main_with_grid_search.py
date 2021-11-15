# This code writes down the solution for Task 1
from parse import parse
import pickle as pkl
from torch.utils import data
from utils.data_utils import Series_Dataset, obtain_tr_val_test_idx, check_if_dir_or_file_exists
from utils.data_utils import get_dataloaders, load_saved_dataset, create_splits_file_name, load_splits_file
import os
import torch
import json
#from src.rnn_models import RNN_model, train_rnn
from src.rnn_models_NLSS_modified import RNN_model, train_rnn
import argparse
from utils.gs_utils import create_list_of_dicts

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python main_with_grid_search.py --data [path to data file] --dataset_type [pfixed/vars/all] --model_type [gru/lstm/rnn]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")

    parser.add_argument("--datafile", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--dataset_mode", help="Enter the type of dataset (pfixed/vars/all)", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    parser.add_argument("--config", help="Enter full path to the configurations json file", type=str)
    parser.add_argument("--splits", help="Enter full path to splits file", type=str)

    args = parser.parse_args() 
    datafile = args.datafile
    dataset_mode = args.dataset_mode
    model_type = args.model_type
    config_file = args.config
    splits_file = args.splits

    # Dataset parameters obtained from the 'datafile' variable
    _, num_trajs, num_realizations, N_seq = parse("{}_M{:d}_P{:d}_N{:d}.pkl", datafile.split('/')[-1])
    batch_size = 128 # Set the batch size

    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'create_function.py' to create the dataset")
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

    batch_size = 128
    train_loader, val_loader, test_loader = get_dataloaders(Z_pM_dataset, batch_size, tr_indices, val_indices, test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))
    
    usenorm_flag = 0
    #for i_batch, sample_batched in enumerate(train_loader):
    #    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    #model_type = "gru"
    #if dataset_mode == "vars":
    #    with open("./config/configurations_var.json") as f:
    #        options = json.load(f)
    #elif dataset_mode == "pfixed":
    #    with open("./config/configurations_alltheta_pfixed.json") as f:
    #        options = json.load(f)
    
    with open(config_file) as f: # Config file for estimating theta_vector when some parameters are fixed
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    logfile_path = "./log/estimate_theta_{}/".format(dataset_mode)
    #modelfile_path = "./models/"

    #NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_L{}_H{}_modified_RNN".format(model_type, options[model_type]["n_layers"], options[model_type]["n_hidden"])

    #print(params)
    # Json file to store grid search results
    jsonfile_name = 'grid_search_results_{}_{}_NS{}.json'.format(model_type, dataset_mode, int(num_trajs*num_realizations))
    gs_log_file_name = "gs_training_modified_RNN_{}_M{}_P{}_N{}.log".format(model_type,
                                                            num_trajs,
                                                            num_realizations,
                                                            N_seq)
    
    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(os.path.join(logfile_path, main_exp_name),
                                                            file_name=gs_log_file_name)
    
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    
    #flag_models_dir, _ = check_if_dir_or_file_exists(os.path.join(modelfile_path, main_exp_name),
    #                                                file_name=None)
    
    #print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))
    
    tr_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), gs_log_file_name)
    jsonfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), jsonfile_name)

    if flag_log_dir == False:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)
    
    # Parameters to be tuned
    if dataset_mode == "vars":
        gs_params = {
                    "n_hidden":[30, 40, 50, 60],
                    "n_layers":[1, 2],
                    "num_epochs":[1000, 2000],
                    "n_hidden_dense":[32, 64]
                    }
    elif dataset_mode == "pfixed":
        gs_params = {
                    "n_hidden":[40, 50, 60, 70],
                    "n_layers":[1, 2],
                    "num_epochs":[3000],
                    "n_hidden_dense":[32, 40]
                    }
    
    # Creates the list of param combinations (options) based on the provided 'model_type'
    gs_list_of_options = create_list_of_dicts(options=options,
                                            model_type=model_type,
                                            param_dict=gs_params)
        
    print("Grid Search to be carried over following {} configs:\n".format(len(gs_list_of_options)))
    val_errors_list = []

    for i, gs_option in enumerate(gs_list_of_options):
        
        # Load the model with the corresponding options
        model = RNN_model(**gs_option)
    
        tr_verbose = True 
        save_chkpoints = None
        tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model = train_rnn(options=gs_option, 
                                                                                            nepochs=gs_option["num_epochs"],
                                                                                            train_loader=train_loader,
                                                                                            val_loader=val_loader,
                                                                                            device=device,
                                                                                            usenorm_flag=usenorm_flag,
                                                                                            tr_verbose=tr_verbose,
                                                                                            logfile_path=tr_logfile_name_with_path,
                                                                                            save_chkpoints=save_chkpoints)
        
        gs_option["Config_no"] = i+1
        gs_option["tr_loss_end"] = tr_losses[-1]
        gs_option["val_loss_end"] = val_losses[-1]
        gs_option["tr_loss_best"] = tr_loss_for_best_val_loss
        gs_option["val_loss_best"] = best_val_loss

        val_errors_list.append(gs_option)
        
    with open(jsonfile_name_with_path, 'w') as f:
        f.write(json.dumps(val_errors_list, indent=2))

    return None


if __name__ == "__main__":
    main()

    
