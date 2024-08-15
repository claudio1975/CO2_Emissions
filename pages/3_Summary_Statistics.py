import numpy as np
import pandas as pd
import streamlit as st


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./CO2_Emissions_Canada.csv')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)


st.title("Summary Statistics")

st.markdown("""
The data set is composed by 12 variables and 7385 rows. 
""")


st.subheader('Categorical Variables')
# Summarize attribute distributions for data type of variables
obj_cols = [var for var in df.columns if df[var].dtype in ['object']]
df[obj_cols].describe().T


st.subheader('Numerical Variables')
# Summarize attribute distributions for data type of variables
num_cols = [var for var in df.columns if df[var].dtype in ['int64','float64']]
df[num_cols].describe().T
