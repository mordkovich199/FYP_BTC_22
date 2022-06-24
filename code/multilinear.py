import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ParameterGrid
import sys
from scipy import stats
import os
import sys
import warnings
import time

from preprocessing import full_dataset, calculate_pca
from preprocessing import preprocess
from models import multilinear, r_squared, predict, plot_learning_curve, update_progress
from feature_selection import get_top_features

warnings.filterwarnings("ignore")
os.environ['WDM_LOG_LEVEL'] = '0'

N_JOBS = 20
VALIDATION_SPLITS = 50

df_base = full_dataset()

def evaluate(pca=False, pca_n=3, outliers_cutoff=2.5, glassnode=False, top_features_n=15, metrics_type="Tier1", mode='optimize'):
    optimize = True if mode == "optimize" else False
    visualize_all = True if mode == "visualize" else False

    if(glassnode):
        x, y = get_top_features(top_features_n, metrics_type, verbal=False)
        x = x.astype(float)
        y = y.astype(float)
        y = y['v']

    splits = np.linspace(0.10, 0.85, VALIDATION_SPLITS)
    test_size=0.10
    total_experiments = N_JOBS * VALIDATION_SPLITS
    experiment_counter = 0

    if(not(optimize)):
        if(glassnode):
            print(f"Train set min size: {round(len(x)*(1-(test_size+max(splits))))}")
            print(f"Train set max size: {round(len(x)*(1-(test_size+min(splits))))}")
        else:
            print(f"Train set min size: {round(len(df_base)*(1-(test_size+max(splits))))}")
            print(f"Train set max size: {round(len(df_base)*(1-(test_size+min(splits))))}")
    # input("Wanna continue?")

    #Pre-processing parameters
    z_score=True

    #Analytical Evaluation
    test_rmses = []
    test_r2s = []

    #Vizualisation and other reasons
    test_models = []
    ma_model = 20
    ma_lc = 5

    #variables used to plot the learning curve I suppose
    train_rmses = []
    train_r2s = []
    val_rmses = []
    val_r2s = []
    train_set_size = []

    #Typical pre-processing in linear problems
    #(1)Take the z-score
    #(2)Remove outliers [e.g. magnitude grater than 2.5] as in completely from the dataset
    #(3)Keep the y dataset unchanged

    
    if(not(glassnode)):
        df = df_base.copy()
        x, y = preprocess(df)
        
    x_valtrain, x_test, y_valtrain, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

    x_test = x_test.apply(lambda cell: stats.zscore(cell))
    if pca: x_test = calculate_pca(x_test, n=pca_n, verbose = False)

    for val_size in splits:

        split_train_rmse_list = []
        split_train_r2_list = []
        split_val_rmse_list = []
        split_val_r2_list = []

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
            
            weights, model =multilinear(x_train, y_train, verbose=False)
            split_models.append(model)

            #Training set performance (Later take the average of error across multiple experiments for the same split)
            # train_rmse = sqrt(mean_squared_error(y_train, df_new['predicted']))
            train_rmse = mean_squared_error(y_train, model.predict(x_train))
            split_train_rmse_list.append(train_rmse)
            train_r2 = r_squared(y_train, model.predict(x_train))
            split_train_r2_list.append(train_r2)

            #Validation set performance
            # val_rmse = sqrt(mean_squared_error(y_val, df_new['predicted']))
            val_rmse = mean_squared_error(y_val, model.predict(x_val))
            split_val_rmse_list.append(val_rmse)
            val_r2 = r_squared(y_val, model.predict(x_val))
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
        split_best_model = split_models[split_val_rmse_list.index(val_best_rmse)]

        #Now we're testing the model on selected parameters under real-life conditions
        test_r2 = r_squared(y_test, split_best_model.predict(x_test))
        test_r2s.append(test_r2)
        # test_rms = sqrt(mean_squared_error(y_test, df_new['predicted']))
        test_rms = mean_squared_error(y_test, split_best_model.predict(x_test))
        test_rmses.append(test_rms)

        test_models.append(split_best_model)
    
    if optimize: return round(sqrt(np.mean(val_rmses))), round(np.mean(val_r2s), 2), \
                        round(sqrt(np.mean(test_rmses))), round(np.mean(test_r2s), 2)
    elif(visualize_all):
        best_test_rmse = min(test_rmses)
        best_test_r2 = max(test_r2s)
        best_model_rmse = test_models[test_rmses.index(best_test_rmse)]
        best_model_r2 = test_models[test_r2s.index(best_test_r2)]

        if(z_score):
            x = x.apply(lambda cell: stats.zscore(cell))

        if pca: x = calculate_pca(x, n=pca_n, verbose = False)

        index = y.index.astype('datetime64[ns]')

        y_predicted_rmse = best_model_rmse.predict(x)
        y_predicted_r2 = best_model_r2.predict(x)
        y_predicted = [(y_predicted_rmse[i]+y_predicted_r2[i])/2 for i in range(len(y_predicted_rmse))]
        y_predicted_ma = [np.nan]*ma_model+[np.mean(y_predicted[i:i+ma_model]) for i in range(len(y_predicted)-ma_model)]

        return index, y, y_predicted_ma
    else:
        print(f"\n\nAverage Validaion RMSE: {round(sqrt(np.mean(val_rmses)))}")
        print(f"Average Validation R^2: {round(np.mean(val_r2s), 2)}")
        print(f"Average Test  RMSE: {round(sqrt(np.mean(test_rmses)))}")
        print(f"Average Test R^2: {round(np.mean(test_r2s), 2)}")
        print(f"Outlier cutoff: {outliers_cutoff}")
        if pca: print(f"PCA n= {pca_n}")

        train_dictionary = dict(zip(train_set_size, train_rmses))
        val_dictionary = dict(zip(train_set_size, val_rmses))
        train_set_size.sort()
        train_rmses = [train_dictionary[elem] for elem in train_set_size]

        train_rmses_ma = [np.nan]*ma_lc+[np.min(train_rmses[i:i+ma_lc]) for i in range(len(train_rmses)-ma_lc)]
        val_rmses = [val_dictionary[elem] for elem in train_set_size]
        val_rmses_ma = [np.nan]*ma_lc+[np.max(val_rmses[i:i+ma_lc]) for i in range(len(val_rmses)-ma_lc)]

        #Plotting the learning curve
        plt.plot(train_set_size, train_rmses, label = "Training")
        plt.plot(train_set_size, val_rmses, label = "Validation")
        # plt.plot(train_set_size, train_rmses_ma, label = "Training MA")
        # plt.plot(train_set_size, val_rmses_ma, label = "Validation MA")

        if pca: 
            name = f"MLR PCA(n={pca_n})" 
        else: 
            name = "MLR no PCA"

        plt.title(f'Learning Curve, {name}')
        plt.xlabel('Training Examples')
        plt.ylabel('MSE')
        plt.legend()

        # difference = [train_rmses[i]-val_rmses[i] for i in range(len(val_rmses))]
        # idx = np.argwhere(np.diff(np.sign(difference))).flatten()
        # plt.plot(train_set_size[idx[0]], val_rmses[idx[0]], color='red', marker='o', markersize=10)

        plt.show()

        #The model visualization part
        best_test_rmse = min(test_rmses)
        best_test_r2 = max(test_r2s)
        best_model_rmse = test_models[test_rmses.index(best_test_rmse)]
        best_model_r2 = test_models[test_r2s.index(best_test_r2)]

        if(z_score):
            x = x.apply(lambda cell: stats.zscore(cell))

        if pca: x = calculate_pca(x, n=pca_n, verbose = False)

        index = y.index.astype('datetime64[ns]')

        y_predicted_rmse = best_model_rmse.predict(x)
        y_predicted_r2 = best_model_r2.predict(x)
        y_predicted = [(y_predicted_rmse[i]+y_predicted_r2[i])/2 for i in range(len(y_predicted_rmse))]
        y_predicted_ma = [np.nan]*ma_model+[np.mean(y_predicted[i:i+ma_model]) for i in range(len(y_predicted)-ma_model)]

        plt.plot(index, y, label = "Actual Price")
        # plt.plot(index, y_predicted, label = "Model Price")
        plt.plot(index, y_predicted_ma, label = f"Model Price MA={ma_model}")

        plt.title(f'Multilinear Model')
        plt.xlabel('Date')
        plt.ylabel('Price, $')
        plt.legend()

        
        plt.show()



# st = time.time()
# evaluate(pca=False, pca_n=3, outliers_cutoff=3.5, glassnode=True, top_features_n=10, metrics_type="Tier1_2", mode='optimize')
# et = time.time()
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')


def optimize():
    pca_Ns = [3, 5, 7, 10, 12, 15, 17, 20]
    cutoffs = [1, 1.5, 2, 2.5, 3, 3.5]
    top_features_ns = [10, 15, 20, 30, 40]
    hyper_grid = {'pca_n': pca_Ns, 'outliers_cutoff': cutoffs, 'top_features_n':top_features_ns}
    grid = list(ParameterGrid(hyper_grid))

    for params in grid:
        evaluate(pca=True, pca_n=params['pca_n'], outliers_cutoff=params['outliers_cutoff'])

    for params in cutoffs:
        evaluate(pca=False, outliers_cutoff=params)

def optimize_advanced():
    OUTPUT = []
    counter = 0
    
    pca_Ns = [3, 5, 7, 10, 12, 15, 17, 20]
    cutoffs = [1, 1.5, 2, 2.5, 3, 3.5]
    top_features_ns = [10, 15, 20, 30, 40]

    #PCA & Tier 1 OR Tier 2
    hyper_grid = {'pca_n': pca_Ns, 'outliers_cutoff': cutoffs, 'top_features_n':top_features_ns}
    grid_pca = list(ParameterGrid(hyper_grid))
    grid_pca_1 = grid_pca.copy()
    grid_pca_12 = grid_pca.copy()

    #No PCA & Tier 1 OR Tier 2
    hyper_grid.pop('pca_n')
    grid_no_pca = list(ParameterGrid(hyper_grid))
    grid_no_pca_1 = grid_no_pca.copy()
    grid_no_pca_12 = grid_no_pca.copy()

    print(f"Number of hyperparameter combinations: {2*len(grid_pca)+2*len(grid_no_pca)}")
    print(f"Estimated execution time: {round((65*(2*len(grid_pca)+2*len(grid_no_pca)))/60/60, 2)} h")
    input("Wanna continue?")

    #Loop 1
    for g_1 in grid_pca_1:
        # print(g_1)
        try:

            # if(grid_pca.index(g_1)==2):
            #     print("Breaking the first loop")
            #     break

            params = list(g_1.values())
            avg_val_rmse, avg_val_r2, avg_test_rmse, avg_test_r2 = evaluate(pca=True, pca_n=g_1['pca_n'], outliers_cutoff=g_1['outliers_cutoff'], glassnode=True, top_features_n=g_1['top_features_n'], metrics_type="Tier1", optimize=True)
            stats = g_1.copy()
            stats['avg_val_rmse']=avg_val_rmse
            stats['avg_val_r2']=avg_val_r2
            stats['avg_test_rmse']=avg_test_rmse
            stats['avg_test_r2']=avg_test_r2
            stats['tier']='Tier 1'
            OUTPUT.append(stats)

            counter += 1
            print(f"{counter} out of {2*len(grid_pca)+2*len(grid_no_pca)}")
        except Exception as e:
            print(f"Error occured: {e}")
            print(params)
        
    #Loop 2
    for g_2 in grid_pca_12:
        # print(g_2)
        try:

            # if(grid_pca.index(g_2)==2):
            #     print("Breaking the second loop")
            #     break

            params = list(g_2.values())
            avg_val_rmse, avg_val_r2, avg_test_rmse, avg_test_r2 = evaluate(pca=True, pca_n=g_2['pca_n'], outliers_cutoff=g_2['outliers_cutoff'], glassnode=True, top_features_n=g_2['top_features_n'], metrics_type="Tier1_2", optimize=True)
            stats = g_2.copy()
            stats['avg_val_rmse']=avg_val_rmse
            stats['avg_val_r2']=avg_val_r2
            stats['avg_test_rmse']=avg_test_rmse
            stats['avg_test_r2']=avg_test_r2
            stats['tier']='Tier 1_2'
            OUTPUT.append(stats)

            counter += 1
            print(f"{counter} out of {2*len(grid_pca)+2*len(grid_no_pca)}")
        except Exception as e:
            print(f"Error occured: {e}")
            print(params)

    #Loop 3
    for g_3 in grid_no_pca_1:
        # print(g_3)
        try:
            # if(grid_no_pca.index(g_3)==2):
            #     print("Breaking the third loop")
            #     break

            params = list(g_3.values())
            avg_val_rmse, avg_val_r2, avg_test_rmse, avg_test_r2 = evaluate(pca=False, outliers_cutoff=g_3['outliers_cutoff'], glassnode=True, top_features_n=g_3['top_features_n'], metrics_type="Tier1", optimize=True)
            stats = g_3.copy()
            stats['avg_val_rmse']=avg_val_rmse
            stats['avg_val_r2']=avg_val_r2
            stats['avg_test_rmse']=avg_test_rmse
            stats['avg_test_r2']=avg_test_r2
            stats['tier']='Tier 1'

            OUTPUT.append(stats)

            counter += 1
            print(f"{counter} out of {2*len(grid_pca)+2*len(grid_no_pca)}")
        except Exception as e:
            print(f"Error occured: {e}")
            print(params)

    #Loop 4
    for g_4 in grid_no_pca_12:
        # print(g_4)
        try:
            # if(grid_no_pca.index(g_4)==2):
            #     print("Breaking the fourth loop")
            #     break

            params = list(g_4.values())
            avg_val_rmse, avg_val_r2, avg_test_rmse, avg_test_r2 = evaluate(pca=False, outliers_cutoff=g_4['outliers_cutoff'], glassnode=True, top_features_n=g_4['top_features_n'], metrics_type="Tier1_2", optimize=True)
            stats = g_4.copy()
            stats['avg_val_rmse']=avg_val_rmse
            stats['avg_val_r2']=avg_val_r2
            stats['avg_test_rmse']=avg_test_rmse
            stats['avg_test_r2']=avg_test_r2
            stats['tier']='Tier 1_2'
            OUTPUT.append(stats)

            counter += 1
            print(f"{counter} out of {2*len(grid_pca)+2*len(grid_no_pca)}")
        except Exception as e:
            print(f"Error occured: {e}")
            print(params)

    print(OUTPUT)
    return OUTPUT


# out = optimize_advanced()
# results_table = pd.DataFrame(out)
# results_table = results_table.replace(np.nan, "na")
# # results_table.to_csv("results.csv", mode='a', header=False)
# results_table.to_csv("results_multireg_new.csv")

#-------------------------------------------------------------------------------------------

# optimize()


# df = df_base.copy()
# x, y = preprocess(df)

# if(z_score):
#     x = x.apply(lambda cell: stats.zscore(cell))
#     x = x[(np.abs(x) < 2.5).all(axis = 1)]
#     y = y.loc[x.index]

# if pca: x = calculate_pca(x, n=pca_n, verbose = False)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

# actual_val_sizes = [elem/(1-test_size) for elem in splits]
# train_sizes = [1-elem for elem in actual_val_sizes]
# # actual_train_sizes = [int(len(x_valtrain)*elem) for elem in train_sizes]

# # print(train_set_size)
# # print(actual_train_sizes)

# train_sizes, train_scores, validation_scores = learning_curve(
# estimator = LinearRegression(),
# X = x_train,
# y = y_train, train_sizes = train_sizes, cv = 5,
# scoring = 'neg_mean_squared_error',
# shuffle = True)

# train_scores_mean = -train_scores.mean(axis = 1)
# validation_scores_mean = -validation_scores.mean(axis = 1)

# plt.style.use('seaborn')
# plt.plot(train_sizes, train_scores_mean, label = 'Training error')
# plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
# plt.ylabel('MSE', fontsize = 14)
# plt.xlabel('Training set size', fontsize = 14)
# plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
# plt.legend()
# plt.show()