
import pandas as pd
import numpy as np
import env

def get_db_url(database):
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

def get_data_from_sql():
    query = """
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id = 3;
    """
    df = pd.read_sql(query, get_db_url('telco_churn'))
    return df

def wrangle_telco():
    """
    Queries the telco_churn database
    Returns a clean df with four columns:
    customer_id(object), monthly_charges(float), tenure(int), total_charges(float)
    """
    customers = get_data_from_sql()
    #customers = pd.read_sql("SELECT customer_id, monthly_charges, tenure, total_charges FROM customers WHERE contract_type_id = 3", env.get_db_url('telco_churn'))
    customers['total_charges'] = customers['total_charges'].str.strip()
    customers = customers.replace(r'^\s*$', np.nan, regex=True)
    customers = customers.dropna()
    customers['total_charges'] = customers['total_charges'].astype(float)
    return customers
    

#print(customers.sort_values(by='total_charges'))
#print(f' Shape = {customers.shape}')
#print(f'Describe = {customers.describe()}')