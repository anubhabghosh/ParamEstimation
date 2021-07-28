from create_dataset_revised import create_and_save_dataset, create_filename
import os
from utils.gs_utils import create_combined_param_dict
from parse import parse

def get_list_of_datasets(dataset_mode, dataset_path, param_dict_dataset, create_folders=False, use_norm=0):

    ##############################################################################################
    # Modify this part before running the code
    ##############################################################################################
    
    usenorm_flag = use_norm
    mode = dataset_mode
    output_path = dataset_path

    #param_dict_dataset = {"N_seq":[200],
    #         "num_trajectories": [400, 500, 600],
    #         "num_realizations": [50, 100, 200]
    #         }

    ##############################################################################################

    # Creates the list of param combinations (dataset_options) 
    params_combination_list = create_combined_param_dict(param_dict_dataset)

    # Creates the list of param combinations (dataset_options) 
    list_of_datasets = []

    # Creating the dataset folders
    for params in params_combination_list:
    
        num_trajectories = params["num_trajectories"]
        num_realizations = params["num_realizations"]
        N_seq = params["N_seq"]

        dataset_folder_name = "M{}_P{}_N{}/".format(num_trajectories,
                                                   num_realizations,
                                                    N_seq)
    
        #print(os.path.join(output_path, dataset_folder_name))
        full_path_data_folder = os.path.join(output_path, dataset_folder_name)

        # Creates the individual folders for storing the datasets
        if create_folders == True:
            try:
                os.makedirs(full_path_data_folder, exist_ok=True)    
            except:
                raise FileExistsError

        # Create the full path for the datafile
        datafile = create_filename(P=num_realizations, M=num_trajectories, N=N_seq, use_norm=usenorm_flag, 
                            dataset_basepath=full_path_data_folder, mode=mode)

        list_of_datasets.append(datafile)

    return list_of_datasets

def main():    

    ##############################################################################################
    # Modify this part before running the code
    ##############################################################################################
    
    usenorm_flag = 0
    mode = "pfixed"
    output_path = "./data/estimate_partialfixed_theta/"
    create_folders_flag=True

    param_dict_dataset = {"N_seq":[200],
             "num_trajectories": [400, 500, 600],
             "num_realizations": [50, 100, 200]
             }

    ##############################################################################################

    list_of_datasets = get_list_of_datasets(dataset_mode=mode,
                                            dataset_path=output_path,
                                            param_dict_dataset=param_dict_dataset,
                                            create_folders=create_folders_flag,
                                            use_norm=usenorm_flag)

    for datafile in list_of_datasets:
    
        # If the dataset hasn't been already created, create the dataset
        if not os.path.isfile(datafile):
            print("Creating the data file: {}".format(datafile))
            idx = datafile.rfind("trajectories")
            mode, num_trajectories, num_realizations, N_seq = parse("trajectories_data_{}_M{:d}_P{:d}_N{:d}",datafile[idx:].split('.')[0])
            create_and_save_dataset(N=N_seq, num_trajs=num_trajectories, num_realizations=num_realizations,
                                    filename=datafile, usenorm_flag=usenorm_flag, mode=mode)
            print("Done...")
        
        else:
            print("{} is already present!".format(datafile))

    return None

if __name__ == "__main__":
    main()







