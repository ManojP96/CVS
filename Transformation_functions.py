import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Eda_functions import format_numbers,line_plot,summary
import numpy as np
import re

def sanitize_key(key, prefix=""):
    # Use regular expressions to remove non-alphanumeric characters and spaces
    key = re.sub(r'[^a-zA-Z0-9]', '', key)
    return f"{prefix}{key}"


def check_box(options, ad_stock_value,lag_value,num_columns=4, prefix=""):
    num_rows = -(-len(options) // num_columns)  # Ceiling division to calculate rows

    selected_options = []
    adstock_info = {}  # Store adstock and lag info for each selected option
    if ad_stock_value!=0:
        for row in range(num_rows):
            cols = st.columns(num_columns)
            for col in cols:
                if options:
                    option = options.pop(0)
                    key = sanitize_key(f"{option}_{row}", prefix=prefix)
                    selected = col.checkbox(option, key=key)
                    if selected:
                        selected_options.append(option)
                        
                        # Input minimum and maximum adstock values
                        adstock = col.slider('Select Adstock Range', 0.0, 1.0, ad_stock_value, step=0.05, format="%.2f",key= f"adstock_{key}" )

                        # Input minimum and maximum lag values
                        lag = col.slider('Select Lag Range', 0, 7, lag_value, step=1,key=f"lag_{key}" )

                        # Create a dictionary to store adstock and lag info for the option
                        option_info = {
                            'adstock': adstock,
                            'lag': lag}
                        # Append the dictionary to the adstock_info list
                        adstock_info[option]=option_info

                    else:adstock_info[option]={
                            'adstock': ad_stock_value,
                            'lag': lag_value}

        return selected_options, adstock_info   
    else:
        for row in range(num_rows):
            cols = st.columns(num_columns)
            for col in cols:
                if options:
                    option = options.pop(0)
                    key = sanitize_key(f"{option}_{row}", prefix=prefix)
                    selected = col.checkbox(option, key=key)
                    if selected:
                        selected_options.append(option)
                        
                        # Input minimum and maximum lag values
                        lag = col.slider('Select Lag Range', 0, 7, lag_value, step=1,key=f"lag_{key}" )

                        # dictionary to store adstock and lag info for the option
                        option_info = {
                            'lag': lag}
                        # Append the dictionary to the adstock_info list
                        adstock_info[option]=option_info

                    else:adstock_info[option]={
                            'lag': lag_value}
                        
        return selected_options, adstock_info 

def apply_lag(X, features,lag_dict):
    #lag_data=pd.DataFrame()
    for col in features:
        for lag in range(lag_dict[col]['lag'][0], lag_dict[col]['lag'][1] + 1):
            if lag>0:
                X[f'{col}_lag{lag}'] = X[col].shift(periods=lag, fill_value=0)
    return X

def apply_adstock(X, variable_name, decay):
    values = X[variable_name].values
    adstock = np.zeros(len(values))
    
    for row in range(len(values)):
        if row == 0:
            adstock[row] = values[row]
        else:
            adstock[row] = values[row] + adstock[row - 1] * decay
    
    return adstock

def top_correlated_features(df,target,media_data):
    corr_df=df.drop(target,axis=1)
    #corr_df[target]=df[target]
    #st.dataframe(corr_df)
    for i in media_data:
        #st.write(media_data[2])
        #st.dataframe(corr_df.filter(like=media_data[2]))
        d=(pd.concat([corr_df.filter(like=i),df[target]],axis=1)).corr()[target]
        d=d.sort_values(ascending=False)
        d=d.drop(target,axis=0)
        corr=pd.DataFrame({'Feature_name':d.index,"Correlation":d.values})
        corr.columns = pd.MultiIndex.from_product([[i], ['Feature_name', 'Correlation']])

    return corr

def top_correlated_features(df,variables,target):
    correlation_df=pd.DataFrame()
    for col in variables:
        d=pd.concat([df.filter(like=col),df[target]],axis=1).corr()[target]
        #st.dataframe(d)
        d=d.sort_values(ascending=False).iloc[1:]
        corr_df=pd.DataFrame({'Media_channel':d.index,'Correlation':d.values})
        corr_df.columns=pd.MultiIndex.from_tuples([(col, 'Variable'), (col, 'Correlation')])
        correlation_df=pd.concat([corr_df,correlation_df],axis=1)
    return correlation_df

def top_correlated_feature(df,variable,target):
    d=pd.concat([df.filter(like=variable),df[target]],axis=1).corr()[target]
    # st.dataframe(d)
    d=d.sort_values(ascending=False).iloc[1:]
    # st.dataframe(d)
    corr_df=pd.DataFrame({'Media_channel':d.index,'Correlation':d.values})
    corr_df['Adstock']=corr_df['Media_channel'].map(lambda x:x.split('_adst')[1] if len(x.split('_adst'))>1 else '-')
    corr_df['Lag']=corr_df['Media_channel'].map(lambda x:x.split('_lag')[1][0] if len(x.split('_lag'))>1 else '-' )
    corr_df.drop(['Correlation'],axis=1,inplace=True)
    corr_df['Correlation']=np.round(d.values,2)
    sorted_corr_df= corr_df.loc[corr_df['Correlation'].abs().sort_values(ascending=False).index]
    #corr_df.columns=pd.MultiIndex.from_tuples([(variable, 'Variable'), (variable, 'Correlation')])
    #correlation_df=pd.concat([corr_df,correlation_df],axis=1)
    return sorted_corr_df