import pandas as pd
import os
from env import username, host, password 

# Acquiring telco_churn data
def get_connection(db, username=username, host=host, password=password):
    '''
    Creates a connection URL
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'


def new_telco_churn_data():
    '''
    Returns telco_churn into a dataframe
    '''
    sql_query = '''select * from customers
    join internet_service_types using(internet_service_type_id)
    join contract_types using(contract_type_id)
    join payment_types using(payment_type_id)'''
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    return df 


def get_telco_churn_data():
    '''get connection, returns telco_churn into a dataframe and creates a csv for us'''
    if os.path.isfile('telco_churn.csv'):
        df = pd.read_csv('telco_churn.csv', index_col=0)
    else:
        df = new_telco_churn_data()
        df.to_csv('telco_churn.csv')
    return df
    