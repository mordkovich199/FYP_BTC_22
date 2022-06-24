import json
import requests
import pandas as pd
import calendar
from datetime import datetime
import time
import urllib3

avoid_errors = (requests.exceptions.ConnectionError, urllib3.exceptions.MaxRetryError, urllib3.exceptions.NewConnectionError, TimeoutError)


# insert your API key here
API_KEY = '2273Y0heYa7Yr4lJeaIbjjmvA2U'
core_url = 'https://api.glassnode.com'
available_tier_max = 1
start_date = '2011-08-18'
#I've added 1 day to what you will see as the last row in the old table, because glassnode's until date,
#doesn't mean including.
end_date = '2021-12-02'
date_format = '%Y-%m-%d'

date_since = datetime.strptime(start_date, date_format)
date_since = calendar.timegm(date_since.utctimetuple())

date_to = datetime.strptime(end_date, date_format)
date_to = calendar.timegm(date_to.utctimetuple())

#Collecting all available unique metrics
res = requests.get(core_url + '/v2/metrics/endpoints',
    params={'api_key': API_KEY})

result = res.text
metrics_list = eval(result.split('\n')[0])
metrics_list = [elem for elem in metrics_list if elem['tier']<=available_tier_max]
metrics_list = [elem for elem in metrics_list if True in [x['symbol']=='BTC' for x in elem['assets']]]
metrics_list = [elem for elem in metrics_list if '24h' in elem['resolutions']]
metrics_list = [elem['path'] for elem in metrics_list]


def get_full_dataset(metrics_list):
    TABLE = pd.DataFrame()
    for mq in metrics_list:
        print(f"{metrics_list.index(mq)} out of {len(metrics_list)}")
        try:
            res = requests.get(core_url + mq,
                params={'api_key': API_KEY, 'a':'BTC', 's':date_since, 'u':date_to, 'i':'24h'})
        except avoid_errors:
            print(f"Something went wrong with http request - avoiding this metric {mq}")
            continue
        
        mq_name = mq.split('/')
        mq_name = mq_name[len(mq_name)-1]
        print(mq_name + " | " + str(res.status_code))


        if(res.status_code != 200):
            print(f"Something is wrong - the query {mq_name} didn't work: {res.text}")
            continue
        
        if(res.text == 'null' or res.text == '[]'):
            print(f"Something is wrong - the query {mq_name} returned 'null'")
            continue
        
        #This is a list of dictionaries
        try:
            data = eval(res.text)
        except Exception:
            text = res.text
            text = text.replace('null', "'"+"nan"+"'")
            data = eval(text)

        try:
            check_item = data[0]['v']
        except Exception:
            dic_keys = list(data[0].keys())
            v_name = dic_keys[len(dic_keys)-1]

            print(f"Weird metric: {mq_name}")
            print(data[0])
            decision = input("Either write the name of the key or type SKIP to avoid this metric\n")
            if(decision=='SKIP'):
                continue
            else:
                # dic_keys_2 = list(data[0][v_name].keys())
                # v_name_2 = dic_keys_2[0]
                data = [{'t':elem['t'], 'v':elem[v_name][decision]} for elem in data]

        # print(f"Dataset length: {len(data)}")
        #Because we want to keep the new dataset exactly the same length as the dataset we used previously
        #for all experiments to achieve consistency.
        if(len(data)<3759):
            print("Small dataset")
            continue

        df = pd.DataFrame.from_dict(data)
        df = df.rename(columns={"v": mq_name})
        if(TABLE.empty):
            TABLE = df
        else: 
            TABLE = TABLE.merge(df, how='outer', on='t')
        
        # print(f"Table length: {len(TABLE)}")
        time.sleep(5)
    
    TABLE['index'] = TABLE['t'].apply(lambda x: datetime.fromtimestamp(x).strftime(date_format))
    TABLE.index = list(TABLE['index'].values)
    TABLE.drop('t', axis=1, inplace=True)
    TABLE.drop('index', axis=1, inplace=True)

    return TABLE


TABLE = get_full_dataset(metrics_list)
TABLE.to_csv("Glassnode_Tier1.csv")
