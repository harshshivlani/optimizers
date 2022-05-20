#nbi:hide_in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import edhec_risk_kit as erk
import edhec_risk_kit_206 as erk1
from datetime import date
import work
from datetime import date, timedelta
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import base64

import plotly.express as px
import plotly.graph_objects as go
import plotly

#Starters

#Export To Excel Buttons

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="NAV_Data.xlsx">Export to Excel</a>' # decode b'abc' => abc



st.write("""
# Portfolio Optimization Demo
""")

cust_df = st.file_uploader('Choose an Excel File')
if cust_df:
    cust_df = pd.read_excel(cust_df, header=0, index_col=0, engine='openpyxl').dropna()

    #Max Sharpe Optimzier Function 
    import scipy
    from scipy.optimize import minimize

    def msr(prices, riskfree_rate=0, method='SLSQP'):
        n = prices.shape[1] #no of assets, since er row headers will be the number of assets
        init_guess = np.repeat(1/n, n) 
        bounds = ((0.0, 1.0),)*n
        weights_sum_to_1 = {'type':'eq','fun': lambda weights: np.sum(weights)-1}

        def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
                """
                Returns the negative of the sharpe ratio given the weights
                """
                r = weights.T @ er
                vol = (weights.T @ cov @ weights)**0.5
                return -(r - riskfree_rate/252)/vol

        er = prices.pct_change().mean()
        cov = np.array(prices.pct_change().cov())

        result = minimize(neg_sharpe_ratio, init_guess,
                               args= (riskfree_rate, er, cov), method = method,   #Args means additional arguments required i.e. covariance matrix here for the portfolio vol function above #"SLSQP" is the method name for "Quadratic Optimization"
                               bounds=bounds,                 #Bounds define minimum and maximum weights as we defined above
                               constraints=(weights_sum_to_1,),
                               options={'disp':False}
                              )
        return result.x

    #Max Sharpe Backtester Function
    @st.cache(allow_output_mutation=True)
    def sharpe_bt(prices, rf_rate=0, rebalancing='Quarterly', lookback=90):
        """
        """
        df = prices.copy()

        #convert daily data to monthly/quarterly
        if rebalancing=='Monthly':
            period1 = pd.Series(df.index.month)
            period2 = pd.Series(df.index.month).shift(-1)
        elif rebalancing=='Quarterly':
            period1 = pd.Series(df.index.quarter)
            period2 = pd.Series(df.index.quarter).shift(-1)
        elif rebalancing=='Yearly':
            period1 = pd.Series(df.index.year)
            period2 = pd.Series(df.index.year).shift(-1)
        elif rebalancing=='Weekly':
            period1 = pd.Series(df.index.week)
            period2 = pd.Series(df.index.week).shift(-1)        
            
        mask = (period1 != period2)
        rebal = df[mask.values]
        weights = pd.DataFrame(index=df.columns)

        #calculate % weights based on Max Sharpe Allocation
        for i in range(len(rebal)):
            end = rebal.index[i]
            start = end - timedelta(days=lookback)
            weights = weights.join(pd.DataFrame(msr(df[start:end], rf_rate/100, 'SLSQP'), index=df.columns, columns=[rebal.index[i]]))

        weights= weights.T
        weights.index.name='Date'
        weights.columns = weights.columns+'(%)'

        #Merge weights data with prices data (monthly/quarterly)
        newdf = df.merge(weights, on='Date').dropna()
        newdf['MSR Portfolio']=np.nan
        newdf['MSR Portfolio'][0] = 10000
        newdf[df.columns+'(Q)']=0

        newdf.iloc[0,-6:] = np.divide((newdf['MSR Portfolio'][0]*newdf[df.columns+'(%)'].iloc[0,:]).values, newdf[df.columns].iloc[0,:].values)
        for i in range(1,len(newdf)):
            newdf['MSR Portfolio'][i] = (newdf[df.columns+'(Q)'].iloc[i-1,:].values*newdf[df.columns].iloc[i,:].values).sum()
            newdf.iloc[i,-6:] = np.divide((newdf['MSR Portfolio'][i]*newdf[df.columns+'(%)'].iloc[i,:]).values, newdf[df.columns].iloc[i,:].values)

        final = df.join(newdf[df.columns+'(Q)'], on='Date').ffill().round(4).dropna()
        final['MSR Portfolio'] = np.multiply(final[df.columns],final[df.columns+'(Q)']).sum(axis=1)
        final = final.join(newdf[df.columns+'(%)'], on='Date').ffill()
        return final



    with st.expander('Sharpe Ratio Optimization Parameters'):
        d1, d2, d9 = st.columns(3)
        rebal_period = d1.selectbox('Rebalancing Frequency',('Quarterly', 'Monthly'))
        lookback = d2.number_input('Lookback Period (in business days): ', min_value=30, max_value=252*3, value=90, step=1)
        rf_rate = d9.number_input('Enter Risk Free Rate (%, Annualized)', value=0.0, step=0.01, key='optimizer')

    #if st.button("Run Maximum Sharpe Ratio Optimization"):
    port = sharpe_bt(cust_df, rf_rate, rebal_period, lookback)

    #Show Portfolio Allocations Over Time
    st.write("## Max Sharpe Ratio Allocations - "+rebal_period+"  Rebalanced")
    fig1 = px.area(port[cust_df.columns+'(%)'])
    fig1.update_layout(xaxis_title='Date',
                               yaxis_title='Allocation (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                               legend_title_text='Securities', plot_bgcolor = 'White', yaxis_tickformat = '.0%')
    fig1.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(get_table_download_link(port[cust_df.columns+'(%)']), unsafe_allow_html=True)


    st.write("## Performance Chart")
    tri = pd.DataFrame(port['MSR Portfolio']).merge(pd.DataFrame(port[cust_df.columns].mean(axis=1)), on='Date').merge(port[cust_df.columns], on='Date')
    tri.columns=['Max Sharpe Portfolio', 'Equally Weighted Portfolio'] + list(cust_df.columns)


    def plot_chart(tri):
        d5, d6 = st.columns(2)
        start= d5.date_input("Custom Start Date: ", value=date(2020,1,1), min_value=tri.index[0])
        end = d6.date_input("Custom End Date: ", value=tri.index[-1], max_value=tri.index[-1])
        tri = pd.DataFrame((1+tri[start:end].pct_change().dropna(0)).cumprod()-1)
        fig = px.line(tri)
        fig.update_layout(     xaxis_title='Date',
                               yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                               legend_title_text='Portfolios', plot_bgcolor = 'White', yaxis_tickformat = '.0%')
        fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}') 
        fig.update_yaxes(automargin=True, zeroline=True, zerolinecolor='black')
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig
    st.plotly_chart(plot_chart(tri), use_container_width=True)


    st.write("""
    ## Performance Analytics
    """)
    d3, d4 = st.columns(2)
    periodicity = d3.selectbox('Select Data Frequency: ', ('Daily', 'Monthly'))
    rf = d4.number_input('Enter Risk Free Rate (%, Annualized)', value=0.0, step=0.01)

    if periodicity =='Daily':
        freq = 252
    else:
        freq=12

    rets = pd.DataFrame(port['MSR Portfolio']).merge(pd.DataFrame(cust_df.mean(axis=1)), on='Date').merge(cust_df, on='Date').pct_change().dropna()
    rets.columns = ['MSR Portfolio', 'EW Portfolio']+list(cust_df.columns)
    d5, d6 = st.columns(2)
    start= d5.date_input("Custom Start Date: ", value=tri.index[0], min_value=tri.index[0], key='summary')
    end = d6.date_input("Custom End Date: ", value=tri.index[-1], max_value=tri.index[-1], key='summary')
    summary_metrics = erk.summary_stats(rets[start:end], rf/100, freq)
    st.dataframe(summary_metrics)
    st.markdown(get_table_download_link(summary_metrics), unsafe_allow_html=True)