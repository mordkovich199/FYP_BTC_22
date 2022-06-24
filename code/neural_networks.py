from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
import time
import os

from charting import plot
from preprocessing import full_dataset, calculate_pca
from preprocessing import preprocess
from models import r_squared, update_progress

import warnings
warnings.filterwarnings("ignore")

N_JOBS = 10
VALIDATION_SPLITS = 5
df_base = full_dataset()

def evaluate(pca=False, pca_n=3, outliers_cutoff=2.5, hidden_layers=(10, 20), mode='optimize'):
    optimize = True if mode == "optimize" else False
    visualize_all = True if mode == "visualize" else False


    splits = np.linspace(0.10, 0.85, VALIDATION_SPLITS)
    test_size=0.10
    total_experiments = N_JOBS * VALIDATION_SPLITS
    experiment_counter = 0
    if(not(optimize)):
        print(f"Train set min size: {round(len(df_base)*(1-(test_size+max(splits))))}")
        print(f"Train set max size: {round(len(df_base)*(1-(test_size+min(splits))))}")
    # input("Wanna continue?")

    #Pre-processing parameters
    z_score=True

    #Analytical Evaluation
    test_mses = []
    test_r2s = []
    test_accuracies = []

    #Vizualisation and other reasons
    test_models = []
    ma_model = 20
    ma_lc = 5

    #variables used to plot the learning curve I suppose
    train_mses = []
    train_r2s = []
    val_mses = []
    val_r2s = []
    train_set_size = []

    #Typical pre-processing in linear problems
    #(1)Take the z-score
    #(2)Remove outliers [e.g. magnitude grater than 2.5] as in completely from the dataset
    #(3)Keep the y dataset unchanged

    df = df_base.copy()
    x, y = preprocess(df)
    x_valtrain, x_test, y_valtrain, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

    x_test = x_test.apply(lambda cell: stats.zscore(cell))
    if pca: x_test = calculate_pca(x_test, n=pca_n, verbose = False)

    for val_size in splits:

        split_train_mse_list = []
        split_train_r2_list = []
        split_train_accuracies_list = []
        split_val_mse_list = []
        split_val_r2_list = []
        split_val_accuracies_list = []

        split_models = []

        #Running multiple splits for the same training set size
        for i in range(N_JOBS):
            experiment_counter += 1
            update_progress(experiment_counter/total_experiments, f"Split: {round(val_size, 2)} Experiment: {i}")

            if(z_score):
                x_intermediate = x_valtrain.apply(lambda cell: stats.zscore(cell))
                x_intermediate = x_intermediate[(np.abs(x_intermediate) < outliers_cutoff).all(axis = 1)]
                y_intermediate = y_valtrain.loc[x_intermediate.index]

            if pca: x_intermediate = calculate_pca(x_intermediate, n=pca_n, verbose = False)

            x_train, x_val, y_train, y_val = train_test_split(x_intermediate, y_intermediate, test_size=val_size/(1-test_size), shuffle=True)#test_size is the size of the second variable - in this case x_val and y_val
            
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers)
            model = mlp.fit(x_train, y_train)
            split_models.append(model)

            #Training set performance (Later take the average of error across multiple experiments for the same split)
            # train_rmse = sqrt(mean_squared_error(y_train, df_new['predicted']))
            train_mse = mean_squared_error(y_train, model.predict(x_train))
            split_train_mse_list.append(train_mse)
            train_r2 = r_squared(y_train, model.predict(x_train))
            split_train_r2_list.append(train_r2)

            #Validation set performance
            # val_rmse = sqrt(mean_squared_error(y_val, df_new['predicted']))
            val_mse = mean_squared_error(y_val, model.predict(x_val))
            split_val_mse_list.append(val_mse)
            val_r2 = r_squared(y_val, model.predict(x_val))
            split_val_r2_list.append(val_r2)

        train_set_size.append(len(x_train))

        train_avg_mse = np.mean(split_train_mse_list)
        train_avg_r2 = np.mean(split_train_r2_list)

        val_avg_mse = np.mean(split_val_mse_list)
        val_avg_r2 = np.mean(split_val_r2_list)

        train_mses.append(train_avg_mse)
        train_r2s.append(train_avg_r2)
        val_mses.append(val_avg_mse)
        val_r2s.append(val_avg_r2)


        #Now the testing procedure starts

        val_best_mse = min(split_val_mse_list)
        val_best_r2 = max(split_val_r2_list)
        split_best_model = split_models[split_val_mse_list.index(val_best_mse)]

        #Now we're testing the model on selected parameters under real-life conditions
        test_r2 = r_squared(y_test, split_best_model.predict(x_test))
        test_r2s.append(test_r2)
        # test_rms = sqrt(mean_squared_error(y_test, df_new['predicted']))
        test_mse = mean_squared_error(y_test, split_best_model.predict(x_test))
        test_mses.append(test_mse)

        test_models.append(split_best_model)

    

    if optimize: return round(sqrt(np.mean(val_mses))), round(np.mean(val_r2s), 2), \
                        round(sqrt(np.mean(test_mses))), round(np.mean(test_r2s), 2)
    elif(visualize_all):
        best_test_mse = min(test_mses)
        best_test_r2 = max(test_r2s)
        best_model_rmse = test_models[test_mses.index(best_test_mse)]
        best_model_r2 = test_models[test_r2s.index(best_test_r2)]

        if(z_score):
            x = x.apply(lambda cell: stats.zscore(cell))

        if pca: x = calculate_pca(x, n=pca_n, verbose = False)

        index = y.index

        y_predicted_rmse = best_model_rmse.predict(x)
        y_predicted_r2 = best_model_r2.predict(x)
        y_predicted = [(y_predicted_rmse[i]+y_predicted_r2[i])/2 for i in range(len(y_predicted_rmse))]
        y_predicted_ma = [np.nan]*ma_model+[np.mean(y_predicted[i:i+ma_model]) for i in range(len(y_predicted)-ma_model)]

        return index, y, y_predicted_ma
        
    else:
        print(f"\\n\\nAverage Validaion RMSE: {round(sqrt(np.mean(val_mses)))}")
        print(f"Average Validation R^2: {round(np.mean(val_r2s), 2)}")
        print(f"Average Test  RMSE: {round(sqrt(np.mean(test_mses)))}")
        print(f"Average Test R^2: {round(np.mean(test_r2s), 2)}")
        print(f"Outlier cutoff: {outliers_cutoff}")
        if pca: print(f"PCA n= {pca_n}")

        train_dictionary_mse = dict(zip(train_set_size, train_mses))
        val_dictionary_mse = dict(zip(train_set_size, val_mses))
        train_dictionary_r2 = dict(zip(train_set_size, train_r2s))
        val_dictionary_r2 = dict(zip(train_set_size, val_r2s))
        train_set_size.sort()

        train_mses = [train_dictionary_mse[elem] for elem in train_set_size]
        train_mses_ma = [np.nan]*ma_lc+[np.min(train_mses[i:i+ma_lc]) for i in range(len(train_mses)-ma_lc)]
        val_mses = [val_dictionary_mse[elem] for elem in train_set_size]
        val_mses_ma = [np.nan]*ma_lc+[np.max(val_mses[i:i+ma_lc]) for i in range(len(val_mses)-ma_lc)]
        train_r2s = [train_dictionary_r2[elem] for elem in train_set_size]
        train_r2s_ma = [np.nan]*ma_lc+[np.min(train_r2s[i:i+ma_lc]) for i in range(len(train_r2s)-ma_lc)]
        val_r2s = [val_dictionary_r2[elem] for elem in train_set_size]
        val_r2s_ma = [np.nan]*ma_lc+[np.min(val_r2s[i:i+ma_lc]) for i in range(len(val_r2s)-ma_lc)]
        
        if pca: 
            name = f"MLR PCA(n={pca_n})" 
        else: 
            name = "MLR no PCA"
        
        #Plotting the learning curve
        plt.plot(train_set_size, train_mses, label = "Training")
        plt.plot(train_set_size, val_mses, label = "Validation")
        # plt.plot(train_set_size, train_mses_ma, label = "Training MA")
        # plt.plot(train_set_size, val_mses_ma, label = "Validation MA")
        plt.title(f'Learning Curve, {name}')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

        #Plotting the r2 curve
        plt.plot(train_set_size, train_r2s, label = "Training")
        plt.plot(train_set_size, val_r2s, label = "Validation")
        # plt.plot(train_set_size, train_r2s_ma, label = "Training MA")
        # plt.plot(train_set_size, val_r2s_ma, label = "Validation MA")
        plt.title(f'Accuracy Curve, {name}')
        plt.xlabel('Epoch')
        plt.ylabel('R^2')
        plt.legend()
        plt.show()

        #The model visualization part
        best_test_mse = min(test_mses)
        best_test_r2 = max(test_r2s)
        best_model_rmse = test_models[test_mses.index(best_test_mse)]
        best_model_r2 = test_models[test_r2s.index(best_test_r2)]

        if(z_score):
            x = x.apply(lambda cell: stats.zscore(cell))

        if pca: x = calculate_pca(x, n=pca_n, verbose = False)

        index = y.index

        y_predicted_rmse = best_model_rmse.predict(x)
        y_predicted_r2 = best_model_r2.predict(x)
        y_predicted = [(y_predicted_rmse[i]+y_predicted_r2[i])/2 for i in range(len(y_predicted_rmse))]
        y_predicted_ma = [np.nan]*ma_model+[np.mean(y_predicted[i:i+ma_model]) for i in range(len(y_predicted)-ma_model)]

        plt.plot(index, y, label = "Actual Price")
        # plt.plot(index, y_predicted, label = "Model Price")
        plt.plot(index, y_predicted_ma, label = f"Model Price MA={ma_model}")

        plt.title(f'NN Model')
        plt.xlabel('Date')
        plt.ylabel('Price, $')
        plt.legend()

        
        plt.show()


def optimize():
    OUTPUT = []
    counter = 0
    pca_Ns = [3, 5, 7]
    cutoffs = [1, 1.5, 2, 2.5, 3, 3.5]
    hidden_n = 2
    max_nodes = 100
    min_nodes = 20
    nodes_step = 20
    nn_configuration = [list(range(min_nodes, max_nodes+nodes_step, nodes_step)) for i in range(hidden_n)]

    hyper_grid = {}
    for i in range(len(nn_configuration)):
        hyper_grid[f'l{i}'] = nn_configuration[i]
    hyper_grid['pca_n'] = pca_Ns
    hyper_grid['outlier_cutoff'] = cutoffs


    grid_full = list(ParameterGrid(hyper_grid))

    hyper_grid.pop('pca_n')
    grid_no_pca = list(ParameterGrid(hyper_grid))

    print(f"Number of hyperparameter combinations: {len(grid_full)+len(grid_no_pca)}")
    input("Wanna continue?")

    for g in grid_full:
        try:

            # if(grid_full.index(g)==10):
            #     break

            params = list(g.values())
            avg_val_rmse, avg_val_r2, avg_test_rmse, avg_test_r2 = evaluate(pca=True, pca_n=params[hidden_n+1], outliers_cutoff=params[hidden_n], hidden_layers=params[0:hidden_n])
            stats = g
            stats['avg_val_rmse']=avg_val_rmse
            stats['avg_val_r2']=avg_val_r2
            stats['avg_test_rmse']=avg_test_rmse
            stats['avg_test_r2']=avg_test_r2
            OUTPUT.append(stats)

            counter += 1
            print(f"{counter} out of {len(grid_full)+len(grid_no_pca)}")
        except Exception as e:
            print(f"Error occured: {e}")
            print(params)

    for g in grid_no_pca:
        try:
            # if(grid_no_pca.index(g)==10):
            #     break

            params = list(g.values())
            avg_val_rmse, avg_val_r2, avg_test_rmse, avg_test_r2 = evaluate(pca=False, outliers_cutoff=params[hidden_n], hidden_layers=params[0:hidden_n])
            stats = g
            stats['avg_val_rmse']=avg_val_rmse
            stats['avg_val_r2']=avg_val_r2
            stats['avg_test_rmse']=avg_test_rmse
            stats['avg_test_r2']=avg_test_r2
            OUTPUT.append(stats)

            counter += 1
            print(f"{counter} out of {len(grid_full)+len(grid_no_pca)}")
        except Exception as e:
            print(f"Error occured: {e}")
            print(params)
    
    print(OUTPUT)
    return OUTPUT

# st = time.time()
evaluate(pca=False, outliers_cutoff=3, hidden_layers=(100, 100), mode='optimize')
# et = time.time()
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')

# out = optimize()
# results_table = pd.DataFrame(out)
# results_table = results_table.replace(np.nan, "na")
# # results_table.to_csv("results.csv", mode='a', header=False)
# results_table.to_csv("results.csv")