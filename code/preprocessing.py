import pandas as pd
from datetime import datetime, timedelta
import sys
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import json
import os
from selenium import webdriver
from charting import save_df_as_image
from charting import plot

EPOCH = datetime.strptime("2005-12-21", "%Y-%m-%d")
date_formats = ['%d.%m.%Y', '%Y-%m-%dT00:00:00Z', '%m/%d/%Y 00:00:00', '%Y-%m-%d']
sp_name = "../Raw Data/sp-price-yahoo.csv"
cdd_name = "../Raw Data/btc-cdd-glassnode.csv"
hashr_name = "../Raw Data/btc-hashrate-glassnode.csv"
nzero_name = "../Raw Data/btc-nzero-addresses-glassnode.csv"
supply_name = "../Raw Data/btc-supply-glassnode.csv"
btc_mcap_name = "../Raw Data/btc-mcap-nasdaq.csv"
tot_mcap_name = "../Raw Data/crypto-mcap-nomics.csv"
us_elec_name = "../Raw Data/us-electricity-price-fred.csv"
btc_price_name = "../Raw Data/btc-price-glassnode.csv"

target_files_default = [sp_name, cdd_name, hashr_name, nzero_name, supply_name, btc_mcap_name, tot_mcap_name, us_elec_name, btc_price_name]
target_columns_default = ['Close', 'cdd-90d', 'value', 'value', '463-day-sf', 'Value', 'value', 'VALUE', 'value']
target_names_default = ['sp500', 'cdd', 'hash_rate', 'nz_addresses', 'sf', 'btc_mcap', 'tot_mcap', 'electricity_cost', 'btc_price']
main_format = '%d.%m.%Y'

def caldate2marketdate(caldate=None):
    """
    Converts from standard datetime instance to integers since epoch.
    """
    if caldate is None:
        caldate = datetime.now()

    linear_day = caldate.weekday()
    linear_day = min(linear_day, 5)
    linear_week = (caldate - EPOCH).days // 7

    return linear_week * 5 + linear_day

def format(dates): 
    global main_format
    """
    dates - list of dates in the original format 'str'
    returns a list of datetime objects 
    """
    for format in date_formats:
        try:        
            dates = [datetime.strptime(date, format) for date in dates]
            main_format = format
            return dates
        except Exception as e:
            print(e)
            continue
    return 0
    
def check_gaps(dates):
    """
    dates - list of datetime objects
    """
    for i in range(len(dates)-1): 
        current = dates[i]
        next = dates[i+1]
        difference = (next-current).days
        if(difference > 1):
            print("Current: " + str(current) + ", Next: " + str(next))
            input("Exposed")
    return 1

def fill_gaps(df, dates):
    i = 0
    while (i!=len(dates)-2):
        i += 1
        print(i)
        current = dates[i]
        next = dates[i+1]
        difference = (next-current).days
        print(difference)
        if(difference > 1):
            print("Current: " + str(current) + ", Next: " + str(next))
            input("Beh")
            # EOW = df.iloc[dates.index(current)]['Close']
            new_date = current + timedelta(days=1) 
            caldate = new_date.strftime(main_format)
            index = dates.index(current)
            line = pd.DataFrame(df.iloc[index].to_dict(), index=[caldate])
            df = pd.concat([df.iloc[:index+1], line, df.iloc[index+1:]])
            dates = list(df.index)
            dates = format(dates)
        
    return df


def full_dataset(target_files=target_files_default, target_columns=target_columns_default, target_names=target_names_default):
    os.chdir("/Users/grigorijmordkovic/Desktop/FYP/code")
    first_dates = []
    last_dates = []
    for i in range(len(target_files)):
        dataset = pd.read_csv(target_files[i], sep=",", index_col=0)
        dates = list(dataset.index)
        dates = format(dates)
        first_dates.append(dates[0])
        last_dates.append(dates[len(dates)-1])

    EPOCH = max(first_dates) #this is a datetime object
    end_date = min(last_dates)
    
    df_out = pd.DataFrame()
    for i in range(len(target_files)):
        dataset = pd.read_csv(target_files[i], sep=",", index_col=0)
        dates = list(dataset.index)
        dates = format(dates)
        caldates = [elem.strftime(main_format) for elem in dates]
        dataset.index = dates
        index_start = dates.index(EPOCH)
        index_end = dates.index(end_date)
        dataset = dataset.iloc[index_start:index_end+1]
        target_column = target_columns[i]
        name = target_names[i]
        df_out[name] = dataset[target_column]
    df_out = df_out.dropna()
    return df_out

def preprocess(df):
    df['btc_dom'] = df['btc_mcap'] / df['tot_mcap']
    df.drop('btc_mcap', axis = 1, inplace=True)
    df.drop('tot_mcap', axis = 1, inplace=True)
    
    y=pd.DataFrame()
    y['btc_price'] = df['btc_price']
    y = y['btc_price']

    columns = list(df.columns)
    columns.remove('btc_price')
    x = df[columns] 
    
    return x, y


def top(x, cap):
    if(x>cap):
        return cap
    else:
        return x

def calculate_pca(df, n, cap = None, verbose = True):
    index = list(df.index)
    pca = PCA(n_components=n)
    x = df.values
    results = pca.fit_transform(x)
    cols = ['pc' + str(i+1) for i in range(n)]
    results = pd.DataFrame(results, columns = cols)
    if(verbose):
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        print("Total explained: {}".format(sum(pca.explained_variance_ratio_)))
    results.index = index
    return results

def fill_empty_values(values):
    for i in range(len(values)):
        val = values[i]
        if(str(val)=='nan'):
            if(i>=1):
                average = (values[i-1] + values[i+1])/2
                values[i] = average
            elif(i==0):
                values[i] = values[i+1]
            else:
                values[i] = values[i-1]
    return values

def calculate_flow(supply, days=463):
    flow=[np.nan]*days
    for i in range(days, len(supply)):
        if (i==0):
            flow.append(0)
        else:
            fl = supply[i]-supply[i-days]
            flow.append(fl)
    return flow

def dataset_visualize(df):
    df.index = [datetime.strftime(date, "%d.%m.%Y") for date in list(df.index)]
    x, y = preprocess(df, z=False)
    x.index.name = "Date"
    x['btc_dom'] = x['btc_dom'].apply(lambda x: round(x, 2))
    x['sf'] = x['sf'].apply(lambda x: round(x, 2))
    x['sp500'] = x['sp500'].apply(lambda x: int(x))
    x['hash_rate'] = x['hash_rate'].apply(lambda x: int(x))
    x['cdd'] = x['cdd'].apply(lambda x: int(x))
    
    chromedriver = '/Users/grigorijmordkovic/Desktop/Crypto/Tech/code/chromedriver'
    driver = webdriver.Chrome(chromedriver) 
    save_df_as_image(x, '/Users/grigorijmordkovic/Desktop/Crypto/Tech/code/dataset.png', driver)

def dataset_smooth(df):
    df['hash_rate'] = df['hash_rate'].rolling(window=20).mean()
    df['btc_dom'] = df['btc_dom'].rolling(window=30).mean()
    df['sp500'] = df['sp500'].rolling(window=10).mean()
    df.dropna(inplace=True)
    return df



# os.chdir('/Users/grigorijmordkovic/Desktop/Crypto/Tech/code')
# dataset = pd.read_csv(cdd_name, sep=",", index_col=0)


# print(list(dataset.values))







