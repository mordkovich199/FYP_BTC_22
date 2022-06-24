from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import os
import pandas as pd
# %matplotlib inline
def Nmaxelements(list_values, list_keys, N):
    final_values = []
    final_keys = []
  
    for i in range(0, N): 
        max1 = 0
          
        for j in range(len(list_values)):     
            if list_values[j] > max1:
                max1 = list_values[j];
                max_key = list_keys[j];
                index_max = j
                  
        list_values.remove(max1)
        list_keys.remove(max_key)
        final_values.append(max1)
        final_keys.append(max_key)
          
    return final_values, final_keys

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_top_features(N, metrics_type='Tier1', verbal=True):
    '''
    metrics_type: 'Tier1' or 'Tier12'
    '''
    if(verbal):
        print(f"Getting top {N} features...")
    prev = os.getcwd()
    os.chdir("/Users/grigorijmordkovic/Desktop/FYP")

    if(metrics_type=='Tier1'):
        X = pd.read_csv("Glassnode_Tier1.csv", index_col=0)
    else:
        X = pd.read_csv("Glassnode_Tier12.csv", index_col=0)

    Y = pd.read_csv("Glassnode_BTC_price.csv", index_col=0)
    os.chdir(prev)

    importances = mutual_info_regression(X, Y)
    features = list(X.columns)
    importances_N, features_N = Nmaxelements(list(importances), features, N)

    df_x = X[features_N]
    if(verbal):
        print("Gotem")
    return df_x, Y


# feat_importances = pd.Series(importances_N_12, features_N_12)
# feat_importances.plot(kind='barh', color='teal', fontsize=8)
# plt.show()
