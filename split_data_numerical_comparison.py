# Libraries
# import libraries

import numpy as np
import pandas as pd
import time
import random
import torch
from torch import nn
import yaml
from utils import *
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    startT = time.time()

    # Experiment parameters
    seed_choice = True
    seed_number = 1
    #robot_choice_list = ['7DoF-7R-Panda', '7DoF-GP66']  # '7DoF-7R-Panda', '7DoF-GP66'
    robot_choice_list = ["6DoF-6R-Jaco", "6DoF-6R-Puma560", "6DoF-6R-Mico", "6DoF-6R-IRB140", "6DoF-6R-KR5",
                "6DoF-6R-UR10", "6DoF-6R-UR3", "6DoF-6R-UR5", "6DoF-6R-Puma260", "6DoF-2RP3R-Stanford",
                "7DoF-7R-Jaco2", "7DoF-7R-Panda", "7DoF-7R-WAM", "7DoF-7R-Baxter", "7DoF-7R-Sawyer",
                "7DoF-7R-KukaLWR4+", "7DoF-7R-PR2Arm", "7DoF-7R-PA10", "7DoF-7R-Gen3", "7DoF-2RP4R-GP66+1"]

    for robot_choice in robot_choice_list:
        print("Splitting for {}:".format(robot_choice))
        #for i in range(1,2):
        for i in range(1,2):
            data_path = '../docker/datasets/'+robot_choice.split('-')[0]+'-Combined/review_data_'+robot_choice+'_1000000_qlim_scale_10_seq_'+str(i)+'.csv'
            
            # Load data
            data = pd.read_csv(data_path)
            data_a = np.array(data)

            # Seed random generators ensure reproducibilities if seed is set to true
            if seed_choice:   
                random.seed(seed_number)
                np.random.seed(seed_number)
                torch.manual_seed(seed_number)
                torch.cuda.manual_seed(seed_number)
                torch.backends.cudnn.deterministic = True


            # Split the dataset as it was done during training to retrieve exactly the same values for fair comparison
            X = data_a[:,:6]
            y = data_a[:,6:]
            
            X_train, X_validate, y_train, y_validate = train_test_split(X, 
                                                                        y, 
                                                                        test_size = 0.1,
                                                                        random_state = 1)

            X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                                y_train, 
                                                                test_size = 0.1,
                                                                random_state = 1)

            # Concatenate X_test and y_test
            data_test = np.concatenate([X_test, y_test], axis=1)
            print('Shape of test data: ', data_test.shape)  
        
            # save test sets 
            if "6DoF" in robot_choice:
                data_test_header = ["x_p", "y_p", "z_p", "R_p", "P_p", "Y_p", "t1_p", "t2_p", "t3_p", "t4_p", "t5_p", "t6_p","x_c", "y_c", "z_c", "R_c", "P_c", "Y_c", "t1_c", "t2_c", "t3_c", "t4_c", "t5_c", "t6_c"]
            elif "7DoF" in robot_choice:
                data_test_header = ["x_p", "y_p", "z_p", "R_p", "P_p", "Y_p", "t1_p", "t2_p", "t3_p", "t4_p", "t5_p", "t6_p", "t7_p","x_c", "y_c", "z_c", "R_c", "P_c", "Y_c", "t1_c", "t2_c", "t3_c", "t4_c", "t5_c", "t6_c", "t7_c"]
            
            df = pd.DataFrame(data_test)
            df.to_csv("../docker/test-datasets/review_data_"+robot_choice+"_1000000_qlim_scale_10_seq_"+str(i)+"_test.csv",
                index=False,
                header=data_test_header)  
            

            endT = time.time()
            print('Elapsed time: {} seconds'.format(endT-startT))
            

        

    

