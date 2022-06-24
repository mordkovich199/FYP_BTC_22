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

os.chdir("/Users/grigorijmordkovic/Desktop/FYP/code")
from multilinear import evaluate as evaluate_mlr
from neural_networks import evaluate as evaluate_nn
from s2f_optimisation import visualize as visualize_s2f
from metcalfeslaw import visualize as visualize_mc

index_mc, y_mc, y_pred_mc = visualize_mc(mode="return")
index_s2f, y_s2f, y_pred_s2f = visualize_s2f(mode="return")
index_mlr, y_mlr, y_pred_mlr = evaluate_mlr(pca=False, pca_n=3, outliers_cutoff=3.5, glassnode=False, top_features_n=15, metrics_type="Tier1", mode='visualize')
index_nn, y_nn, y_pred_nn = evaluate_nn(pca=False, outliers_cutoff=3, hidden_layers=(100, 100), mode='visualize')

plt.plot(index_mlr, y_pred_mlr, label = "Multi-linear", linewidth=1)
plt.plot(index_mlr, y_mlr, label = "BTC price")
plt.plot(index_mlr, y_pred_mc, label = "Metcalf's", linewidth=1)
plt.plot(index_mlr, y_pred_nn, label = "NN", linewidth=1)
plt.plot(index_mlr, y_pred_s2f, label = "S2F", linewidth=1)

plt.legend()
plt.show()