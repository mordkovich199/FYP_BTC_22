from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import os
import sys
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

from charting import plot
from preprocessing import full_dataset, calculate_pca
from preprocessing import preprocess
from math import sqrt


def multilinear(x, y, verbose=True):
    mlr = LinearRegression()  
    mlr.fit(x, y)
    if(verbose):
        print("Intercept: ", mlr.intercept_)
        print("Coefficients: {}".format(list(zip(x, mlr.coef_))))
    return list(zip(x, mlr.coef_)), mlr

def r_squared(in_1, in_2):
    correlation_matrix = np.corrcoef(in_1, in_2)
    correlation_1_2 = correlation_matrix[0,1]
    r_squared = correlation_1_2**2
    return r_squared

def optimize_s2f(df, include_plot=False, f1_lower=0.15, f1_upper=0.40, f1_step=0.05, f2_lower=1.5, f2_upper=5, f2_step=0.01):
    f1s = list(np.arange(f1_lower, f1_upper, f1_step))
    f2s = list(np.arange(f2_lower, f2_upper, f2_step))
    param_grid = {'f1':f1s, 'f2':f2s}
    grid = [elem for elem in ParameterGrid(param_grid)]
    
    r2s = []
    rmss = []
    df_compare = pd.DataFrame()

    for g in grid:
        i = grid.index(g)
        print(f"{i} out of {len(grid)}")
        df_compare[f's2f_{i}'] = round(g['f1'], 2) * pow(df['sf'], round(g['f2'], 2))
        df_compare['actual'] = df['btc_price']
        r2 = r_squared(df_compare[f's2f_{i}'], df_compare['actual'])
        r2s.append(r2)
        rms = mean_squared_error(df_compare['actual'], df_compare[f's2f_{i}'], squared=False)
        rmss.append(rms)
    
    best_r2 = max(r2s)
    best_r2_i = r2s.index(best_r2)
    best_r2_factor = grid[best_r2_i]
    # print(f"Best R^2: {round(best_r2, 2)*100}%")
    # print(f"Best R^2 Factor: {best_r2_factor}")
    # print(f"Average R^2: {round(np.mean(r2s), 2)}")
    df_compare = df_compare.rename(columns={f's2f_{best_r2_i}' : 's2f_r2'})

    best_rms = min(rmss)
    best_rms_factor = grid[rmss.index(best_rms)]
    print(f"Best RMS: {round(best_rms, 2)}")
    # print(f"Best RMS Factor: {best_rms_factor}")
    print(f"Average RMS: {round(np.mean(rmss), 2)}")
    print(f"Worst RMS: {round(max(rmss), 2)}")
    # df_compare = df_compare.rename(columns={f's2f_{best_rms_i}' : 's2f_rms'})


    if include_plot: plot(df_compare, ['actual', 's2f_r2', 's2f_rms'], 's2f', formatted = True, legend=True)
    return 

def optimize_s2f_new(x_train, y_train, grid):
    
    # rmss = []

    # for g in grid:
    #     s2f = g['f1'] * pow(x_train['sf'], g['f2'])
    #     rms = mean_squared_error(y_train, s2f, squared=False)
    #     rmss.append(rms)

    rmss = [mean_squared_error(y_train, elem['f1'] * pow(x_train['sf'], elem['f2']), squared=False) for elem in grid]

    best_rms = min(rmss)
    best_factors = grid[rmss.index(best_rms)]
    best_factors = {'f1' : round(best_factors['f1'], 2), 'f2' : round(best_factors['f2'], 2)}

    return best_factors

def optimize_metcalfs(x_train, y_train, grid):

    rmss = [mean_squared_error(y_train, np.exp(elem['A'] * x_train['metcalfs_vnoa']), squared=False) for elem in grid]

    best_rms = min(rmss)
    best_factors = grid[rmss.index(best_rms)]
    best_factors = {'A' : round(best_factors['A'], 2)}

    return best_factors

    

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best")
    plt.show()
    # return plt

def predict(inputs, coefficients):
    coefficients=dict(coefficients)
    data_names = list(coefficients.keys())
    out = 0
    # input(data_names)
    # input(inputs)
    for inp in data_names:
        out += inputs[inp]*coefficients[inp]
    return out

def update_progress(progress, name):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0} | [{1}] {2}%".format(name, "#"*block + "-"*(barLength-block), round(progress*100, 4))
    sys.stdout.write(text)
    sys.stdout.flush()
    


