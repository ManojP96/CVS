import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from collections import OrderedDict
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import re
from matplotlib.colors import ListedColormap
# from st_aggrid import AgGrid, GridOptionsBuilder
# from src.agstyler import PINLEFT, PRECISION_TWO, draw_grid


def format_numbers(x):
    if abs(x) >= 1e6:
        # Format as millions with one decimal place and commas
        return f'{x/1e6:,.1f}M'
    elif abs(x) >= 1e3:
        # Format as thousands with one decimal place and commas
        return f'{x/1e3:,.1f}K'
    else:
        # Format with one decimal place and commas for values less than 1000
        return f'{x:,.1f}'

    

def line_plot(data, x_col, y1_cols, y2_cols, title):
    fig = go.Figure()
      
    for y1_col in y1_cols:
        fig.add_trace(go.Scatter(x=data[x_col], y=data[y1_col], mode='lines', name=y1_col,line=dict(color='#11B6BD')))

    for y2_col in y2_cols:
        fig.add_trace(go.Scatter(x=data[x_col], y=data[y2_col], mode='lines', name=y2_col, yaxis='y2',line=dict(color='#739FAE')))
    if len(y2_cols)!=0:
        fig.update_layout(yaxis=dict(), yaxis2=dict(overlaying='y', side='right'))
    else:
        fig.update_layout(yaxis=dict(), yaxis2=dict(overlaying='y', side='right'))
    if title:
        fig.update_layout(title=title)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def line_plot_target(df,target,title):

    coefficients = np.polyfit(df['Date'].view('int64'), df[target], 1)
    trendline = np.poly1d(coefficients)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date'], y=df[target], mode='lines', name=target,line=dict(color='#11B6BD')))
    trendline_x = df['Date']
    trendline_y = trendline(df['Date'].view('int64'))


    fig.add_trace(go.Scatter(x=trendline_x, y=trendline_y, mode='lines', name='Trendline', line=dict(color='#739FAE')))

    fig.update_layout(
        title=title,
        xaxis=dict(type='date')
    )

    for year in df['Date'].dt.year.unique()[1:]:

        january_1 = pd.Timestamp(year=year, month=1, day=1)
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=january_1,
                x1=january_1,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="grey", width=1.5, dash="dash"),
            )
        )

    return fig

def correlation_plot(df,selected_features,target):
    custom_cmap = ListedColormap(['#08083B', "#11B6BD"])  
    corr_df=df[selected_features]
    corr_df=pd.concat([corr_df,df[target]],axis=1)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr_df.corr(),annot=True, cmap='Blues', fmt=".2f", linewidths=0.5,mask=np.triu(corr_df.corr()))
    #plt.title('Correlation Plot')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    return fig

def summary(data,selected_feature,spends,Target=None):
        
        if Target:
            sum_df = data[selected_feature]
            sum_df['Year']=data['Date'].dt.year
            sum_df=sum_df.groupby('Year')[selected_feature].sum()
            sum_df=sum_df.reset_index()
            total_sum = sum_df.sum(numeric_only=True)
            total_sum['Year'] = 'Total' 
            sum_df = sum_df.append(total_sum, ignore_index=True)
            sum_df.set_index(['Year'],inplace=True)
            sum_df=sum_df.applymap(format_numbers)
            # sum_df = sum_df.reindex(sorted(sum_df.columns), axis=1)

            return sum_df
        else:
            #selected_feature=list(selected_feature)
            selected_feature.append(spends) 

            sum_df = data[selected_feature]
            sum_df['Year']=data['Date'].dt.year
            sum_df=sum_df.groupby('Year')[selected_feature].agg('sum')
            sum_df['CPM/CPC']=(sum_df.iloc[:, 1] / sum_df.iloc[:, 0])*1000
            sum_df.loc['Grand Total']=sum_df.sum()
          
            sum_df=sum_df.applymap(format_numbers)
            sum_df.fillna('-',inplace=True)
            sum_df=sum_df.replace({"0.0":'-','nan':'-'})
            return sum_df



def sanitize_key(key, prefix=""):
    # Use regular expressions to remove non-alphanumeric characters and spaces
    key = re.sub(r'[^a-zA-Z0-9]', '', key)
    return f"{prefix}{key}"




