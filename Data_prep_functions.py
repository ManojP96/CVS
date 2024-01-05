import streamlit as st
import pandas as pd
import re
from collections import OrderedDict
import pandas as pd
import plotly.graph_objs as go

def calculate_discount(promo_price_series, non_promo_price_series):
    # Calculate the 4-week moving average of non-promo price
    window_size = 4
    base_price = non_promo_price_series.rolling(window=window_size).mean()
    
    # Calculate discount_raw
    discount_raw_series = (1 - promo_price_series / base_price) * 100
    
    # Calculate discount_final
    discount_final_series = discount_raw_series.where(discount_raw_series >= 5, 0)
    
    return base_price, discount_raw_series, discount_final_series


def create_dual_axis_line_chart(date_series, promo_price_series, non_promo_price_series, base_price_series, discount_series):
    # Create traces for the primary axis (price vars)
    trace1 = go.Scatter(
        x=date_series,
        y=promo_price_series,
        name='Promo Price',
        yaxis='y1'
    )
    
    trace2 = go.Scatter(
        x=date_series,
        y=non_promo_price_series,
        name='Non-Promo Price',
        yaxis='y1'
    )

    trace3 = go.Scatter(
        x=date_series,
        y=base_price_series,
        name='Base Price',
        yaxis='y1'
    )
    
    # Create a trace for the secondary axis (discount)
    trace4 = go.Scatter(
        x=date_series,
        y=discount_series,
        name='Discount',
        yaxis='y2'
    )

    # Create the layout with dual axes
    layout = go.Layout(
        title='Price and Discount Over Time',
        yaxis=dict(
            title='Price',
            side='left'
        ),
        yaxis2=dict(
            title='Discount',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        xaxis=dict(title='Date'),
    )

    # Create the figure with the defined traces and layout
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    return fig


