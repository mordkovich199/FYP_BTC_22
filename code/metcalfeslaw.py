from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import os
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import warnings

from charting import plot
from preprocessing import full_dataset, calculate_pca
from preprocessing import preprocess
from preprocessing import sp_name, cdd_name, hashr_name, nzero_name, supply_name, btc_mcap_name, tot_mcap_name, us_elec_name, btc_price_name
from math import sqrt
from models import optimize_s2f_new, r_squared, update_progress, optimize_metcalfs

warnings.filterwarnings("ignore")
os.environ['WDM_LOG_LEVEL'] = '0'

N_JOBS = 20
VALIDATION_SPLITS = 50

files = [sp_name, cdd_name, hashr_name, nzero_name, supply_name, supply_name, btc_mcap_name, tot_mcap_name, us_elec_name, btc_price_name]
columns = ['Close', 'cdd-90d', 'value', 'value', 'stock', '463-day-sf', 'Value', 'value', 'VALUE', 'value']
names = ['sp500', 'cdd', 'hash_rate', 'nz_addresses', 'supply', 'sf', 'btc_mcap', 'tot_mcap', 'electricity_cost', 'btc_price']
df_base = full_dataset(target_files=files, target_columns=columns, target_names=names)

def Nmaxelements(list1, N):
    final_list = []
  
    for i in range(0, N): 
        max1 = 0
          
        for j in range(len(list1)):     
            if list1[j] > max1:
                max1 = list1[j];
                  
        list1.remove(max1);
        final_list.append(max1)
          
    print(final_list)

def Nminelements(list1, N):
    final_list = []
  
    for i in range(0, N): 
        min1 = 1
          
        for j in range(len(list1)):     
            if list1[j] < min1:
                min1 = list1[j];
                  
        list1.remove(min1);
        final_list.append(min1)
          
    print(final_list)

def predict_metcalf(inputs, factors):
    A = factors['A']
    out = np.exp(A * inputs['metcalfs_vnoa'])
    return out

def preprocess_for_metcalfs(x):
    x['metcalfs_d'] = x['nz_addresses'].apply(lambda x: (x*(x-1))/(2*pow(10, 6)))
    x['metcalfs_e'] = x['supply'].apply(lambda x: (x*np.log(21*pow(10, 6)/x))/pow(10, 6))
    x['metcalfs_vnoa'] = x.apply(lambda x: np.log(x.metcalfs_d)/x.metcalfs_e, axis=1)

    return x


def evaluate():

    splits = np.linspace(0.10, 0.85, VALIDATION_SPLITS)
    test_size=0.10
    total_experiments = N_JOBS * VALIDATION_SPLITS
    experiment_counter = 0
    print(f"Train set min size: {round(len(df_base)*(1-(test_size+max(splits))))}")
    print(f"Train set max size: {round(len(df_base)*(1-(test_size+min(splits))))}")

    #Analytical Evaluation
    test_rmses = []
    test_r2s = []

    #Used to plot the learning curve
    train_rmses = []
    train_r2s = []
    val_rmses = []
    val_r2s = []
    train_set_size = []

    #Used for visualization and other purposes
    test_factors = []

    #Input parameters
    A_lower=0.05
    A_upper=1.50
    A_step=0.05
    As = list(np.arange(A_lower, A_upper, A_step))
    param_grid = {'A':As}
    grid = [elem for elem in ParameterGrid(param_grid)]

    df = df_base.copy()
    x, y = preprocess(df)
    # y = np.log(y)
    x = preprocess_for_metcalfs(x)
    x_valtrain, x_test, y_valtrain, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

    for val_size in splits:

        split_train_rmse_list = []
        split_train_r2_list = []
        split_val_rmse_list = []
        split_val_r2_list = []

        split_factors_list = []

        for i in range(N_JOBS):
            experiment_counter += 1
            update_progress(experiment_counter/total_experiments, f"Split: {round(val_size, 2)} Experiment: {i}")

            x_train, x_val, y_train, y_val = train_test_split(x_valtrain, y_valtrain, test_size=val_size/(1-test_size), shuffle=True)#test_size is the size of the second variable - in this case x_val and y_val
        
            factors = optimize_metcalfs(x_train, y_train, grid)
            split_factors_list.append(factors)

            #Training set performance
            train_rmse = mean_squared_error(y_train, predict_metcalf(x_train, factors))
            split_train_rmse_list.append(train_rmse)
            train_r2 = r_squared(y_train, predict_metcalf(x_train, factors))
            split_train_r2_list.append(train_r2)

            #Validation set performance
            val_rmse = mean_squared_error(y_val, predict_metcalf(x_val, factors))
            split_val_rmse_list.append(val_rmse)
            val_r2 = r_squared(y_val, predict_metcalf(x_val, factors))
            split_val_r2_list.append(val_r2)
        
        
        train_set_size.append(len(x_train))

        train_avg_rmse = np.mean(split_train_rmse_list)
        train_avg_r2 = np.mean(split_train_r2_list)

        val_avg_rmse = np.mean(split_val_rmse_list)
        val_avg_r2 = np.mean(split_val_r2_list)

        train_rmses.append(train_avg_rmse)
        train_r2s.append(train_avg_r2)
        val_rmses.append(val_avg_rmse)
        val_r2s.append(val_avg_r2)

        #Now the testing procedure starts
        val_best_rmse = min(split_val_rmse_list)
        val_best_r2 = max(split_val_r2_list)
        split_best_factors = split_factors_list[split_val_rmse_list.index(val_best_rmse)]

        #Now we're testing the model on selected parameters under real-life conditions
        test_r2 = r_squared(y_test, predict_metcalf(x_test, split_best_factors))
        test_r2s.append(test_r2)
        test_rms = mean_squared_error(y_test, predict_metcalf(x_test, split_best_factors))
        test_rmses.append(test_rms)
        test_factors.append(split_best_factors)
                        
    print(f"\n\nAverage Validaion RMSE: {round(sqrt(np.mean(val_rmses)))}")
    print(f"Average Validation R^2: {round(np.mean(val_r2s), 2)}")
    print(f"Average Test  RMSE: {round(sqrt(np.mean(test_rmses)))}")
    print(f"Average Test R^2: {round(np.mean(test_r2s), 2)}")

    best_test_rmse = min(test_rmses)
    best_factors_rmse = test_factors[test_rmses.index(best_test_rmse)]
    best_test_r2 = max(test_r2s)
    best_factors_r2 = test_factors[test_r2s.index(best_test_r2)]

    print(f"Best Test RMSE Factors: {best_factors_rmse}")
    print(f"Best Test R^2 Factors: {best_factors_r2}")

    train_dictionary = dict(zip(train_set_size, train_rmses))
    val_dictionary = dict(zip(train_set_size, val_rmses))
    train_set_size.sort()
    train_rmses = [train_dictionary[elem] for elem in train_set_size]

    ma_n = 5
    train_rmses_ma = [np.nan]*ma_n+[np.mean(train_rmses[i:i+ma_n]) for i in range(len(train_rmses)-ma_n)]
    val_rmses = [val_dictionary[elem] for elem in train_set_size]
    val_rmses_ma = [np.nan]*ma_n+[np.mean(val_rmses[i:i+ma_n]) for i in range(len(val_rmses)-ma_n)]

    #Plotting the learning curve
    # plt.plot(train_set_size, train_rmses, label = "Training")
    # plt.plot(train_set_size, val_rmses, label = "Validation")
    plt.plot(train_set_size, train_rmses_ma, label = "Training MA")
    plt.plot(train_set_size, val_rmses_ma, label = "Validation MA")
    plt.title(f'Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('MSE')
    plt.legend()

    # difference = [train_rmses[i]-val_rmses[i] for i in range(len(val_rmses))]
    # idx = np.argwhere(np.diff(np.sign(difference))).flatten()
    # plt.plot(train_set_size[idx[0]], val_rmses[idx[0]], color='red', marker='o', markersize=10)

    plt.show()


def visualize(mode='simple'):
    visualize_all = True if mode == "return" else False

    df = df_base.copy()
    x, y = preprocess(df)
    x = preprocess_for_metcalfs(x)

    y_predicted_rmse = predict_metcalf(x, {'A': 1.1})
    y_predicted_r2 = predict_metcalf(x, {'A': 1.1})
    y_predicted = [(y_predicted_rmse[i]+y_predicted_r2[i])/2 for i in range(len(y_predicted_rmse))]
    # y_predicted_ma = [np.nan]*ma_model+[np.mean(y_predicted[i:i+ma_model]) for i in range(len(y_predicted)-ma_model)]

    if(visualize_all):
        return y.index, y, y_predicted
    else:
        plt.plot(y.index, y, label = "Actual Price")
        plt.plot(y.index, y_predicted, label = "Model Price")
        # plt.plot(index, y_predicted_ma, label = f"Model Price MA={ma_model}")

        plt.title(f'Metcalfs')
        plt.xlabel('Date')
        plt.ylabel('Price, $')
        plt.legend()

        plt.show()

# visualize()

# st = time.time()
# evaluate()
# et = time.time()
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')


    
    