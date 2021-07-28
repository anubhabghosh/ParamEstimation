# This code writes down the solution for Task 1
import sys
from utils.data_utils import Series_Dataset, obtain_tr_val_test_idx
from utils.data_utils import get_dataloaders, load_saved_dataset, load_splits_file, NDArrayEncoder
from utils.data_utils import create_splits_file_name
from utils.gs_utils import create_combined_param_dict
import pickle as pkl 
import os
import torch
import json
from src.rnn_models import RNN_model, train_rnn, evaluate_rnn
import argparse
from create_dataset_multiple import get_list_of_datasets

def create_file_paths(params_combination_list, filepath, main_exp_name):
    
    list_of_logfile_paths = []
    # Creating the logfiles
    for params in params_combination_list:

        exp_folder_name = "trajectories_M{}_P{}_N{}/".format(params["num_trajectories"],
                                                            params["num_realizations"],
                                                            params["N_seq"])

        #print(os.path.join(log_filepath, main_exp_name, exp_folder_name))
        full_path_exp_folder = os.path.join(filepath, main_exp_name, exp_folder_name)
        list_of_logfile_paths.append(full_path_exp_folder)
        #os.makedirs(full_path_exp_folder, exist_ok=True)
    
    return list_of_logfile_paths

def check_if_dir_or_file_exists(file_path, file_name=None):
    flag_dir = os.path.exists(file_path)
    if not file_name is None:
        flag_file = os.path.isfile(os.path.join(file_path, file_name))
    else:
        flag_file = None
    return flag_dir, flag_file

def main():

    usage = "use an RNN for parameter estimation using trajectories of non-linear models \n"\
        "python main_run_multiple_datasets.py --mode [train/test]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode (train/test)", type=str)
    parser.add_argument("--model_type", help="Enter the desired model (gru/lstm/rnn)", type=str)
    parser.add_argument("--dataset_mode", help="Enter the type of dataset (pfixed/vars/all)", type=str)
    parser.add_argument("--dataset_path", help="Enter the basepath to the dataset", type=str, default=None)
    parser.add_argument("--model_file_saved", help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str, default=None)
    parser.add_argument("--use_norm", help="Use_normalization", type=int, default=None)
    parser.add_argument("--splits", help="ENter path to splits file", type=int, default=None)
    args = parser.parse_args() 
    mode = args.mode
    dataset_mode = args.dataset_mode
    model_type = args.model_type
    dataset_path = args.dataset_path
    usenorm_flag = args.use_norm
    model_file_saved = args.model_file_saved
    splits_file = args.splits

    param_dict_dataset = {"N_seq":[200],
             "num_trajectories": [400, 500, 600],
             "num_realizations": [50, 100, 200]
             }

    list_of_datasets = get_list_of_datasets(dataset_mode=dataset_mode,
                                            dataset_path=dataset_path,
                                            param_dict_dataset=param_dict_dataset,
                                            create_folders=False,
                                            use_norm=usenorm_flag)

    params_combination_list = create_combined_param_dict(param_dict_dataset)

    #with open("./config/configurations.json") as f: # Config file for estimating all theta parameters
    #    options = json.load(f)

    with open("./config/configurations_alltheta_pfixed.json") as f: # Config file for estimating theta_vector when some parameters are fixed
        options = json.load(f)

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    logfile_path = "./log/estimate_theta_{}/".format(dataset_mode)
    modelfile_path = "./models/"
    main_exp_name = "gru_L{}_H{}_multiple".format(options[model_type]["n_layers"], options[model_type]["n_hidden"])

    list_of_logfile_paths = create_file_paths(params_combination_list=params_combination_list,
                                            filepath=logfile_path,
                                            main_exp_name=main_exp_name)

    list_of_modelfile_paths = create_file_paths(params_combination_list=params_combination_list,
                                            filepath=modelfile_path,
                                            main_exp_name=main_exp_name)

    for i, params in enumerate(params_combination_list):

        datafile = list_of_datasets[i]

        #print(params)
        log_file_name = "training_{}_M{}_P{}_N{}.log".format(model_type,
                                                            params["num_trajectories"],
                                                            params["num_realizations"],
                                                            params["N_seq"])
        
        flag_log_dir, flag_log_file = check_if_dir_or_file_exists(list_of_logfile_paths[i],
                                                                file_name=log_file_name)
        
        print("Is log-directory present:? - {}".format(flag_log_dir))
        print("Is log-file present:? - {}".format(flag_log_file))
        
        flag_models_dir, _ = check_if_dir_or_file_exists(list_of_modelfile_paths[i],
                                                        file_name=None)
        
        print("Is model-directory present:? - {}".format(flag_models_dir))
        #print("Is file present:? - {}".format(flag_file))
        
        if flag_log_dir == False:
            print("Creating {}".format(list_of_logfile_paths[i]))
            os.makedirs(list_of_logfile_paths[i], exist_ok=True)
        
        if flag_models_dir == False:
            print("Creating {}".format(list_of_modelfile_paths[i]))
            os.makedirs(list_of_modelfile_paths[i], exist_ok=True)

        if not os.path.isfile(datafile):
            
            print("{} is not present, run 'create_dataset_revised.py' to create the dataset".format(datafile))
            sys.exit(1)
            #plot_trajectories(Z_pM, ncols=1, nrows=10)
        else:

            print("Dataset already present")
            Z_pM = load_saved_dataset(filename=datafile)
            Z_pM_dataset = Series_Dataset(Z_pM_dict=Z_pM)

        datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))
        #splits_filename_base = os.path.join(datafolder, "splits_file.pkl")
        #splits_file = create_splits_file_name(dataset_filename=datafile,
        #                                    splits_filename=splits_filename_base
        #                                    )
        if splits_file is None or not os.path.isfile(splits_file):
            splits_filename_base = os.path.join(datafolder, "splits_file.pkl")
            splits_file = create_splits_file_name(dataset_filename=datafile,
                                                splits_filename=splits_filename_base
                                                )
            tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_pM_dataset,
                                                                        tr_to_test_split=0.9,
                                                                        tr_to_val_split=0.833)
            print(len(tr_indices), len(val_indices), len(test_indices))
            splits = {}
            splits["train"] = tr_indices
            splits["val"] = val_indices
            splits["test"] = test_indices
            
            with open(splits_file, 'wb') as handle:
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
        
        log_file_path = os.path.join(list_of_logfile_paths[i], log_file_name)
        model_file_path = list_of_modelfile_paths[i]

        #NOTE: This is just to test the main function is working or not!
        #options[model_type]["n_hidden"] = 5
        #options[model_type]["num_epochs"] = 10

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
                                                                                                save_chkpoints="some",
                                                                                                logfile_path=log_file_path,
                                                                                                modelfile_path=model_file_path)
            #if tr_verbose == True:
            #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)
            
            losses_model = {}
            losses_model["tr_losses"] = tr_losses
            losses_model["val_losses"] = val_losses

            with open('./plot_data/{}_losses_eps{}.json'.format(model_type, options[model_type]["num_epochs"]), 'w') as f:
                f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

        elif mode.lower() == "test":

            test_log_file_name = "testing_{}_M{}_P{}_N{}.log".format(model_type,
                                                            params["num_trajectories"],
                                                            params["num_realizations"],
                                                            params["N_seq"])

            #model_file_saved = "./model_checkpoints/{}_usenorm_{}_ckpt_epoch_{}.pt".format(model_type, usenorm_flag, epoch_test)
            evaluate_rnn(options[model_type], test_loader, device, model_file=model_file_saved, usenorm_flag=usenorm_flag,
                        test_logfile_path=os.path.join(list_of_logfile_paths[i], test_log_file_name))
        
    return None


if __name__ == "__main__":
    main()

    
