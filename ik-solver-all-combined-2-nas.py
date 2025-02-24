# Libraries
# import libraries
#from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.utils.data as data
import numpy as np
import pandas as pd
import random
import sklearn
import time
import math
#import matplotlib.pyplot as plt
#import os
import sys
import wandb
import yaml
import optuna
from optuna.trial import TrialState

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from scipy import stats
#from torchviz import make_dot
#from transformers import get_scheduler
from utils import *
from models import *
from models_2 import DenseNet
from model_gpt2 import *
from model_gpt3 import *





# Read parameters from configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--config-path",
                    type=str,
                    default="train.yaml",
                    help="Path to train config file.")
args = parser.parse_args()

with open(args.config_path, "r") as f:
    config = yaml.full_load(f)

#print(config)

    

def train_and_evaluate(trial):

    print('==> Reading from the config file ...')

    # set parameters and configurations
    robot_choice = config["ROBOT_CHOICE"]
    seed_choice = config["SEED_CHOICE"]                                           # seed random generators for reproducibility
    seed_number = config["SEED_NUMBER"]
    print_epoch = config["TRAIN"]["PRINT_EPOCHS"]  
    batch_size = config["TRAIN"]["HYPERPARAMETERS"]["BATCH_SIZE"]                 # desired batch size
    init_type = config["TRAIN"]["HYPERPARAMETERS"]["WEIGHT_INITIALIZATION"]       # weights init method (default, uniform, normal, xavier_uniform, xavier_normal)
    #hidden_layer_sizes = [128,128,128,128]                                       # architecture to employ
    learning_rate = config["TRAIN"]["HYPERPARAMETERS"]["LEARNING_RATE"]           # learning rate
    optimizer_choice = config["TRAIN"]["HYPERPARAMETERS"]["OPTIMIZER_NAME"]       # optimizers (SGD, Adam, Adadelta, RMSprop)
    loss_choice =  config["TRAIN"]["HYPERPARAMETERS"]["LOSS"]                     # l2, l1, lfk
    network_type =  config["MODEL"]["NAME"] 
    num_blocks =  config["MODEL"]["NUM_BLOCKS"]     
    dataset_samples = config["TRAIN"]["DATASET"]["NUM_SAMPLES"]                   # MLP, ResMLP, DenseMLP, FouierMLP 
    print_steps = config["TRAIN"]["PRINT_STEPS"] 
    save_option = config["TRAIN"]["CHECKPOINT"]["SAVE_OPTIONS"]                                # local or cloud
    load_option = config["TRAIN"]["CHECKPOINT"]["LOAD_OPTIONS"]  
    dataset_type = config["TRAIN"]["DATASET"]["TYPE"]
    joint_steps = config["TRAIN"]["DATASET"]["JOINT_VARIATION"]
    orientation_type = config["TRAIN"]["DATASET"]["ORIENTATION"]

    scale = config["TRAIN"]["DATASET"]["JOINT_LIMIT_SCALE"]
    EPOCHS = config["TRAIN"]["HYPERPARAMETERS"]["EPOCHS"]   

    experiments = config["NUM_EXPERIMENT_REPETITIONS"]
    #layers = config["MODEL"]["NUM_HIDDEN_LAYERS"]
    #neurons = config["MODEL"]["NUM_HIDDEN_NEURONS"]                      # total training epochs   

    experiment_number = experiments

    data_path = "/home/datasets"

    print("==> Running based on configuration...")
    device = torch.device('cuda:'+str(config["DEVICE_ID"]) if torch.cuda.is_available() else 'cpu') 
    device_name = torch.cuda.get_device_name(device)
    
    # set input and output size based on robot
    if robot_choice == "All-6DoF":
        if dataset_type == "combine-6DoF":
            n_DoF = 6
            input_dim = 6+6+6
            output_dim = 6

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6"]

    elif robot_choice == "All-7DoF":
        if dataset_type == "combine-7DoF":
            n_DoF = 7
            input_dim = 6+7+6
            output_dim = 7

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]

    elif robot_choice == "All-67DoF":
        if dataset_type == "combine-up-to-7DoF":
            n_DoF = 7
            input_dim = 6+7+6
            output_dim = 7

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
      
       
    if load_option == "cloud":
        if dataset_type == "combine-6DoF":
            
            robot_list = ["6DoF-6R-Jaco", "6DoF-6R-Puma560", "6DoF-6R-Mico", "6DoF-6R-IRB140", "6DoF-6R-KR5", 
                          "6DoF-6R-UR10", "6DoF-6R-UR3", "6DoF-6R-UR5", "6DoF-6R-Puma260", "6DoF-2RP3R-Stanford"]
            

            robot_list_test = ["6DoF-6R-UR10", "6DoF-6R-Puma260"]

            data = np.zeros((dataset_samples, 24, len(robot_list)))           
            for i in range(len(robot_list)):
                df = pd.read_csv(data_path + '/6DoF-Combined/review_data_'+robot_list[i]+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv')
                data[:,:,i] = np.array(df)

        elif dataset_type == "combine-7DoF":

            robot_list = ["7DoF-7R-Jaco2", "7DoF-7R-Panda", "7DoF-7R-WAM", "7DoF-7R-Baxter", "7DoF-7R-Sawyer", 
                          "7DoF-7R-KukaLWR4+", "7DoF-7R-PR2Arm", "7DoF-7R-PA10", "7DoF-7R-Gen3", "7DoF-2RP4R-GP66+1"]
            
            robot_list_test = ["7DoF-7R-WAM", "7DoF-7R-Sawyer"]

            data = np.zeros((dataset_samples, 26, len(robot_list)))           
            for i in range(len(robot_list)):
                df = pd.read_csv(data_path + '/7DoF-Combined/review_data_'+robot_list[i]+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv')
                data[:,:,i] = np.array(df)

        elif dataset_type == "combine-up-to-7DoF":

            
            robot_list_6 = ["6DoF-6R-Jaco", "6DoF-6R-Puma560", "6DoF-6R-Mico", "6DoF-6R-IRB140", "6DoF-6R-KR5", 
                            "6DoF-6R-UR10", "6DoF-6R-UR3", "6DoF-6R-UR5", "6DoF-6R-Puma260", "6DoF-2RP3R-Stanford"]
            
            robot_list_7 = ["7DoF-7R-Jaco2", "7DoF-7R-Panda", "7DoF-7R-WAM", "7DoF-7R-Baxter", "7DoF-7R-Sawyer", 
                            "7DoF-7R-KukaLWR4+", "7DoF-7R-PR2Arm", "7DoF-7R-PA10", "7DoF-7R-Gen3", "7DoF-2RP4R-GP66+1"]
                            
            robot_list = robot_list_6 + robot_list_7
            robot_list_test = ["6DoF-6R-UR10", "6DoF-6R-Puma260", "7DoF-7R-WAM", "7DoF-7R-Sawyer"]

            data = np.zeros((dataset_samples, 26, len(robot_list)))           
            for i in range(len(robot_list)):
                if robot_list[i] in robot_list_6:
                    df = pd.read_csv(data_path + '/6DoF-Combined/review_data_'+robot_list[i]+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv')
                    df['t7_c'] = 0              # Add a column at the end
                    df.insert(12, 't7_p', 0)    # Add a column at column index 12
                elif robot_list[i] in robot_list_7:
                    df = pd.read_csv(data_path + '/7DoF-Combined/review_data_'+robot_list[i]+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv')
                
                data[:,:,i] = np.array(df)
        
    ## train and validate
    # load the dataset
    train_data_loader, test_data_loader, train_test_val_all, sc_in = load_all_dataset_2(data, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim, robot_list, robot_list_test)


    # get network architecture
    if network_type == "MLP":

        ## => Search: 
        ## 1. number of hidden layers
        ## 2. number of hidden neurons
        layers = trial.suggest_int("num_hidden_layers", 1, 10)
        neurons = trial.suggest_categorical("num_hidden_neurons", [128, 256, 512, 768, 1024])
        hidden_layer_sizes = np.zeros((1,layers))          
        hidden_layer_sizes[:,:] = neurons
        hidden_layer_sizes = hidden_layer_sizes.squeeze(0).astype(int).tolist()

        # set the model architecture and saving string
        model = MLP(input_dim, hidden_layer_sizes, output_dim)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
        

    elif network_type == "ResMLP":

        ## ==> Search:
        ## 1. number of residual blocks
        ## 2. number of hidden layers
        ## 3. number of hidden neurons
        #layers = trial.suggest_int("num_hidden_layers", 1, 10)
        neurons = trial.suggest_categorical("num_hidden_neurons", [128, 256, 512, 768, 1024])
        num_blocks = trial.suggest_int("num_residual_blocks", 1, 5)

        model = ResMLPSum(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(2)

    elif network_type == "DenseMLP":
        model = DenseMLP(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)

    elif network_type == "DenseMLP2":
        model = DenseMLP2(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)

    elif network_type == "DenseMLP3":

        ## ==> Search:
        ## 1. number of dense blocks
        ## 2. number of hidden layers
        ## 3. number of hidden neurons

        layers = trial.suggest_int("num_hidden_layers", 1, 5)
        neurons = trial.suggest_categorical("num_hidden_neurons", [128, 256, 512, 768, 1024])
        num_blocks = trial.suggest_int("num_dense_blocks", 1, 5)

        block_config = np.zeros((1,num_blocks))   
        block_config[:,:] = layers
        block_config = block_config.squeeze(0).astype(int).tolist()
        model = DenseNet(input_dim, neurons, block_config, output_dim)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)

    elif network_type == "FourierMLP":
        fourier_dim = 16
        scale = 10
        model = FourierMLP(input_dim, fourier_dim, hidden_layer_sizes, output_dim, scale)

    elif network_type == "GPT2":

        ## ==> Search:
        ## 1. embedding size
        ## 2. number of hidden layers
        ## 3. number of hidden neurons

        embed_dim = trial.suggest_categorical("embed_dim", [96, 192, 384, 768]) #768
        num_head = trial.suggest_categorical("num_heads", [6, 12, 24]) #12
        num_layers = trial.suggest_int("num_hidden_layers", 1, 5)
        neurons = trial.suggest_categorical("num_hidden_neurons", [256, 512, 768, 1024])
        num_blocks = trial.suggest_int("num_residual_blocks", 1, 5)

        num_layers = num_blocks
        model = GPT2ForRegression(input_dim=input_dim, output_dim=output_dim, embed_dim=embed_dim, num_heads=num_head, num_layers=num_layers, ff_dim=neurons)
        save_layers_str = "embed_dim_"+ str(embed_dim)+"_heads_"+ str(num_head)+"_layers_"+ str(num_layers)
    
    elif network_type == "GPT3":
        ## ==> Search:
        ## 1. embedding size
        ## 2. number of hidden layers
        ## 3. number of hidden neurons

        embed_dim = trial.suggest_categorical("embed_dim", [96, 192, 384, 768]) #768
        num_head = trial.suggest_categorical("num_heads", [6, 12, 24]) #12
        num_layers = trial.suggest_int("num_hidden_layers", 1, 5)
        neurons = trial.suggest_categorical("num_hidden_neurons", [256, 512, 768, 1024])
        num_blocks = trial.suggest_int("num_residual_blocks", 1, 5)

        num_layers = num_blocks
        model = GPT3ForRegression(input_dim=input_dim, output_dim=output_dim, embed_dim=embed_dim, num_heads=num_head, num_layers=num_layers, ff_dim=neurons)
        save_layers_str = "embed_dim_"+ str(embed_dim)+"_heads_"+ str(num_head)+"_layers_"+ str(num_layers)
    
    


    model = model.to(device)
    print("==> Architecture: {}\n{}".format(model.name, model))
    print("==> Trainable parameters: {}".format(count_parameters(model)))
    
    # set optimizer
    if optimizer_choice == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_choice == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    elif optimizer_choice == "Adadelta":
        optimizer = optim.Adadelta(model.parameters())
    elif optimizer_choice == "RMSprop":
        optimizer = optim.RMSprop(model.parameters())
    
    # set loss
    if loss_choice == "lq":
        criterion = nn.MSELoss(reduction="mean")
    elif loss_choice == "l1":
        criterion = nn.L1Loss(reduction="mean")
    elif loss_choice == "ld":
        criterion = FKLoss(robot_choice=robot_choice, device=device)


    print("\n==> Experiment {} Training network: {}".format(experiment_number, model.name))
    print("==> Training for joint step {} on device: {}".format(joint_steps, device))
    
   
    save_path = "results_final_combine/"+robot_choice+"/"+network_type+"_"+robot_choice+"_" \
                + save_layers_str + "_neurons_" + str(neurons) + "_batch_" + str(batch_size)  +"_" \
                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)+"_"+dataset_type+"_"+orientation_type+"_"+str(learning_rate)+"_js_"+str(joint_steps)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #if save_option == "cloud":
   

    ##############################################################################################################
    # Training and Validation
    ############################################################################################################## 
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    print("Total training: ", EPOCHS) 
    total_steps = EPOCHS*int(len(train_data_loader))
    
    train_losses = []
    valid_losses = []
    all_losses = []
    best_valid_loss = float('inf')
    start_time_train = time.monotonic()
    start_time = time.monotonic()

    for epoch in range(EPOCHS):        
        
        train_loss = train(model, train_data_loader, optimizer, criterion, loss_choice, batch_size, device, epoch, EPOCHS, scheduler, scaler)        
        valid_loss = evaluate(model, test_data_loader, criterion, loss_choice, device, epoch, EPOCHS)
    
       

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        all_losses.append([train_loss, valid_loss])


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            counter = 0
                   
        if epoch % (EPOCHS/print_steps) == 0 or epoch == EPOCHS-1:
        
            if print_epoch:
                end_time = time.monotonic()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('\nEpoch: {}/{} | Epoch Time: {}m {}s'.format(epoch, EPOCHS, epoch_mins, epoch_secs))
                print('\tTrain Loss: {}'.format(train_loss))
                print('\tValid Loss: {}'.format(valid_loss))
                print("\tBest Epoch Occurred [{}/{}]".format(best_epoch, EPOCHS)) 

            torch.save(model.state_dict(), save_path+'/best_epoch.pth')   
                        
            # save the histories of losses
            
            #header = ["train loss", "valid loss"]
            
            #df = pd.DataFrame(np.array(all_losses))
            #df.to_csv(save_path+"/losses_"+robot_choice+"_"+str(dataset_samples)+".csv",
            #    index=False,
            #    header=header)



        trial.report(valid_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
                      
            
    end_time_train = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time_train, end_time_train)
    
    if print_epoch:
        print('\nEnd of Training for {} - Elapsed Time: {}m {}s'.format(model.name, epoch_mins, epoch_secs))    


    
    


    ##############################################################################################################
    # Inference
    ##############################################################################################################
    # training is done, let's run inferences and record the evaluation metrics
    weights_file = save_path+"/best_epoch.pth"

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)


    inference_results_all = []

    for r in robot_list_test:

        robot_choice = r
        X_test = train_test_val_all[r]["X_test"]
        y_test = train_test_val_all[r]["y_test"]


        # Select 25% of rows (i.e., 2 rows out of 10)
        num_rows = X_test.shape[0]
        subset_size = int(num_rows * 0.25)  # 25% of rows
        random_indices = np.random.choice(num_rows, subset_size, replace=False)

        X_test = X_test[random_indices,:]
        y_test = y_test[random_indices,:]


        print("\n\n==> Testing the trained model on  {} ...".format(r))
        test_data_loader = load_test_all_dataset(X_test, y_test, device, sc_in)
        

        
        
        # get the results from training    
        with torch.no_grad():
            results = inference_modified_all(model, test_data_loader, criterion, device, robot_choice)
        X_errors = results["X_errors_report"]
        
        #print(X_errors.shape)

        # get some inference stats
        X_errors_r = X_errors[:,:6]
        X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
        X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
        avg_position_error = X_errors_r[1,:3].mean()
        avg_orientation_error = X_errors_r[1,3:].mean()

        print("avg_position_error (mm): {}".format(avg_position_error))
        print("avg_orientation_error (deg): {}".format(avg_orientation_error))
        inference_results_all.append(np.mean([avg_position_error, avg_orientation_error]))


    return np.mean(np.array(inference_results_all))


if __name__ == "__main__":

    # Drag and drop the storage file on the browser: https://optuna.github.io/optuna-dashboard/

    print('==> Log in to wandb to send out metrics ...')
    wandb.login()                                        # login to the Weights and Biases   
            


    run = wandb.init(
        entity="jacketdembys",
        project = "ik-steps-2",                
        group = "IROS_25_Search_v20_"+config["MODEL"]["NAME"] +"_"+"Combined_Dataset_"+str(config["TRAIN"]["DATASET"]["NUM_SAMPLES"] )+"_Scale_"+str(int(config["TRAIN"]["DATASET"]["JOINT_LIMIT_SCALE"]))+"_"+config["TRAIN"]["DATASET"]["TYPE"]+"_"+config["TRAIN"]["HYPERPARAMETERS"]["LOSS"],  # "_seq", "_1_to_1"
        name = config["MODEL"]["NAME"]+"_"+config["ROBOT_CHOICE"]+"_" + str(config["TRAIN"]["HYPERPARAMETERS"]["BATCH_SIZE"]) +"_" \
                +config["TRAIN"]["HYPERPARAMETERS"]["OPTIMIZER_NAME"]+"_"+config["TRAIN"]["HYPERPARAMETERS"]["LOSS"]+"_run_"+str(config["NUM_EXPERIMENT_REPETITIONS"])+'_qlim_scale_'+str(int(config["TRAIN"]["DATASET"]["JOINT_LIMIT_SCALE"]))+'_samples_'+str(config["TRAIN"]["DATASET"]["NUM_SAMPLES"] )+"_"+config["TRAIN"]["DATASET"]["ORIENTATION"]+"_"+str(config["TRAIN"]["HYPERPARAMETERS"]["LEARNING_RATE"])+"_js_"+str(config["TRAIN"]["DATASET"]["JOINT_VARIATION"])   #+'_non_traj_split', '_traj_split'   
            )

    # Initialize optuna
    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(study_name=config["ROBOT_CHOICE"]+"_kinematics_search_seq_"+config["MODEL"]["NAME"] ,
                                storage='sqlite:///'+config["ROBOT_CHOICE"]+'_kinematics_search_seq_'+config["MODEL"]["NAME"] +'.db',
                                direction="minimize",
                                sampler=sampler)
    
    study.optimize(train_and_evaluate, n_trials=100, timeout=None) 

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print the best hyperparameters
    best_params = study.best_trial.params
    print("Best trial with hyperparameters found:", best_params)


    all_trials = study.trials_dataframe()
    print(all_trials)

    
  

    
    print(best_params)

    #best_params = pd.concat(best_params)
    table_data = [[key, value] for key, value in best_params.items()]
    results_table = wandb.Table(data=table_data, columns=["Parameter", "Value"])
    wandb.log({"best_params": results_table})


    #wandb.config.update(best_params)
    wandb.finish()




    
    


    