# ParamEstimation
Bayesian estimators for nonlinear system identification by leveraging deep recurrent neural networks.

## Running scripts for Real-world data:
The dataset generation script is create_dataset_ce_drive.py. Example of running it for the PRBS dataset (currently the first input signal `u1` is used for the PRBS dataset generation)
```
python create_dataset_ce_drive.py --num_realizations 100 --num_trajectories 50 --sequence_length 500 --use_norm 0 --output_path ./data/coupled_drive/ --input_signal_path ./data/coupled_drive/DATAPRBS.MAT --input_signal_type prbs
```

The execution script for real-world data is: `main_for_ce_drive.py`. Example of running it:
```
python main_for_ce_drive.py --mode train --model_type gru --data ./data/coupled_drive/ce_drive_trajectories_data_prbs_M50_P100_N500.pkl --use_norm 0
```

The configurations (currently kept the same as `config/configrations_alltheta_pfixed.json`) is stored in `config/configurations_alltheta_ce_drive_prbs.json`
