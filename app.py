import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utilities import set_header,load_local_css,load_authenticator,initialize_data
from Data_prep_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle

st.set_page_config(
    page_title="Sales Simulation",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state='collapsed'
)
# Load CSS and set header (assuming these functions exist)
load_local_css('styles.css')
set_header()
for k, v in st.session_state.items():
    if k not in ['logout', 'login','config'] and not k.startswith('FormSubmitter'):
        st.session_state[k] = v

authenticator = st.session_state.get('authenticator')

if authenticator is None:
    authenticator = load_authenticator()
    
    name, authentication_status, username = authenticator.login('Login', 'main')
    auth_status = st.session_state['authentication_status']

    st.empty()
    data=pd.read_csv('2024_data.csv')
    for col in data:
        if data[col].dtype=='object':
            data[col] = data[col].astype("category")
    
    modeling_features = [
        "market", "geo_segment", "product_segment", "plan_tenure", "samestore_flag", "plantype",
        "avg_prior_2yr_sales_rate",
        "summaryrating_overallplan", "summaryrating_partc_ma", "summaryrating_partd_pdp",
        "stayinghealthy_screenings", "gettingappointmentsquickly", "customerservice",
        "network_index", "hosp_coverage_perc", "pcp_coverage_perc", "spec_coverage_perc",
        "tva_rank", "tva_rank_hmo_ppo", "total_tva", "tva_medical", "tva_drug", "tva_supp",
        "rel_total_tva", "rel_total_tva_hmo_ppo", "rel_tva_drug", "rel_tva_medical",
        "rel_tva_supp", 
        "pref_score_all", "pref_score_hmo_ppo", 
        "predict_ci",
        "average_cost_relative_to_competitor", "tier123_coverage_relative_to_competitor",
        "rural_share",
        "rel_premium", "rel_partbpremiumreduction", "rel_moop", "rel_specialist",
        "rel_deductible", "rel_rxdeductible", "rel_rx_tier1", "rel_rx_tier2",
        "fips_avg_po_basic_premium", "fips_avg_po_enhanced_premium",
        "mapd_hmo_ppo_share", "eligibles", "mapd_penetration", "pdp_penetration",
        "dsnp_penetration", "national_org_market_share", 
        "mapd_plan_count", "parent_org_mapd_market_share", "parent_org_pdp_market_share",
        "parent_org_dsnp_market_share",
        "orig_medicare_penetration",
        "fit_score",
        "res_broker_per_1k_comp_enroll", "total_broker_per_1k_comp_enroll",
        "broker_sales_per_res_broker", "broker_sales_per_total_broker",
        "aet_media_spend_per_1k_comp_enroll", 
        "local_share",
        "phmval300kplus_n", "pemployedbluecollar_n", "ppop16plusunemployed_n", "phispanic_n",
        "pnativeameralone_n", "pfammarriednochild_n", "phhincome100_125k_n",
        "phhincome125_150k_n", "phhincome75_100k_n", "phhspeakspanish_n", "phmval150k_175k_n",
        "phmval300k_400k_n", "phmval70k_100k_n", "phmvalunder50k_n", "pmobilehomehu_n",
        "pnonhsgradage25plus_n", "ptotalwork16plusathome_n", "puptograde8age25plus_n",
        "pmanufacturing_n", "ppublicadmin_n", "ptransportwarehouseutility_n", "punitblt1980_89_n",
        "punitblt1990_99_n", "p2plushu_n", "pemployededucation_n", "pemployedservice_n",
    ]
    
    st.header('Select Market and Product to Proceed')
    
    columns = st.columns(2)
    with columns[0]:
        market=st.selectbox('**Slect Market**',options=list(data['market'].unique()))
    with columns[1]:
        product=st.selectbox('**Select Product**',options=list(data['product'].unique()))
    
    selected_df= data[(data['market']==market) & (data['product']==product)]
    
    
    
    score_X = selected_df[modeling_features]  
    mean_df = score_X.select_dtypes(include=['number']).mean() 
    mean_df = mean_df.to_frame().T 
    mode_df= score_X[['plantype', 'market', 'product_segment']].mode()
    competitor_enroll=selected_df['competitor_enroll'].mean()
    
    # st.write(mean_df)
    # st.write(mode_df)
    final_df=pd.concat([mean_df,mode_df],axis=1)
    final_df=final_df[modeling_features]
    
    # st.write(final_df)
    
    
    # st.write(selected_df.head())
    st.markdown('## Market Summary')
    sales_rate_avg = selected_df['avg_prior_2yr_sales_rate'].mean()
    avg_tenure = selected_df['plan_tenure'].mean()
    mapd_penetration = selected_df['mapd_penetration'].mean()
    
    st.write(f"2yr Sales Rate: {np.round(sales_rate_avg,3)}")
    st.write(f"Avg Plan Tenure: {np.round(avg_tenure,2)}")
    st.write(f"MAPD Penetration: {np.round(mapd_penetration,2)}")
    
    
    st.markdown('#### Adjust the values below, then select "Save" or "Run Simulation".')
    
    
    old_sales=100
    
    def updated_sales():
        new_sales=old_sales+10
        return new_sales
    
    __columns=st.columns(2)
    with __columns[0]:
        ci = st.slider('Competitor Index', min_value=1, max_value=72, step=3, value=int(mean_df["predict_ci"].values))
        tva = st.slider('Total Value Add', min_value=20.0, max_value=350.0, step=20.0, value=float(mean_df["total_tva"].values))
        vsp = st.slider('Value Add Supp', min_value=23.0, max_value=100.0, step=5.0, value=float(mean_df["tva_supp"].values))
    
    with __columns[1]:
        rp = st.slider('Relative Premium', min_value=-100.0, max_value=200.0, step=20.0, value=float(mean_df["rel_premium"].values))
        ms = st.slider('Media Spend', min_value=7.5, max_value=12.5, step=0.5, value=float(mean_df["aet_media_spend_per_1k_comp_enroll"].values))
    
    dtest = xgb.DMatrix(final_df, enable_categorical=True)
    bst = pickle.load(open("xgb_reg.pkl", "rb"))   
    score_Y_pred = bst.predict(dtest)
    predicted_sales_rate_avg=np.exp(score_Y_pred)
    predicted_sales_rate_avg=np.where(predicted_sales_rate_avg < -10, 0, predicted_sales_rate_avg)
    predicted_sales_avg=predicted_sales_rate_avg*competitor_enroll
    predicted_sales_avg=np.round(predicted_sales_avg,2)
    
    
    # st.write(final_df['predict_ci'])
    final_df['predict_ci']=ci 
    final_df["total_tva"]=tva
    final_df['tva_supp']=vsp
    final_df['rel_premium']=rp
    final_df['aet_media_spend_per_1k_comp_enroll']=ms
    # st.write(final_df['predict_ci'])
    dtest = xgb.DMatrix(final_df, enable_categorical=True)
    bst = pickle.load(open("xgb_reg.pkl", "rb"))
    
    
    score_Y_pred = bst.predict(dtest)
    predicted_sales_rate=np.exp(score_Y_pred)
    predicted_sales_rate=np.where(predicted_sales_rate < -10, 0, predicted_sales_rate)
    predicted_sales=predicted_sales_rate*competitor_enroll
    predicted_sales=np.round(predicted_sales,2)
    
    delta_value = float(predicted_sales - predicted_sales_avg)  # Ensure delta_value is a float
    
    
    st.metric('Simulated Sales',f'${predicted_sales[0]:.2f}',delta=f'{delta_value:.2f}')
    
    
    
    
    
    values ={'Competitor Index':mean_df["predict_ci"],'Relative Premium':mean_df["rel_premium"],'Total Value Add':mean_df["total_tva"],
             'Media Spend':mean_df["aet_media_spend_per_1k_comp_enroll"],'Value Add Supp':mean_df["tva_supp"]}
    
    summary_df=pd.DataFrame({'Market':market,'Product':product,'Avg_Competitor_Index':mean_df['predict_ci'],'Sim_Competitor_Index':ci,
                             'Avg_Total_Value_Add':mean_df['total_tva'],"Sim_Total_Value_Add":tva,
                             'Avg_Value_Add_Supp':mean_df['tva_supp'],"Sim_Value_Add_Supp":vsp,
                             'Avg_Relative_Premium':mean_df['rel_premium'],"Sim_Relative_Premium":rp,
                             'Avg_Media_Spend':mean_df['aet_media_spend_per_1k_comp_enroll'],"Sim_Media_Spend":ms,
                             'Sales':predicted_sales_avg,'Simulated_Sales':predicted_sales
                             })
    
    if 'Summary_df' not in st.session_state:
        st.session_state['Summary_df']=pd.DataFrame()
    
    save_columns=st.columns(2)
    with save_columns[0]:
        if st.button("Save Simulation"):
            st.session_state['Summary_df'] = pd.concat([st.session_state['Summary_df'], summary_df])
            st.session_state['Summary_df'].drop_duplicates(inplace=True)
            st.session_state['Summary_df']=st.session_state['Summary_df'].round(2)
    with save_columns[1]:
        if st.button('Clear Simulations'):
            st.session_state['Summary_df']=pd.DataFrame(columns=st.session_state['Summary_df'].columns)
    
    st.dataframe(st.session_state['Summary_df'],use_container_width=True)
