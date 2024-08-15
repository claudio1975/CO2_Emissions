import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro
import streamlit as st

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./data/CO2_Emissions_Canada.csv')

# Delete duplicate rows
df.drop_duplicates(inplace=True)

# Delete duplicate columns
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
# drop useless columns
df.drop(to_del, axis=1, inplace=True)


# Rename some features for a practical use
df_1 = df.rename(columns={
      "Vehicle Class":"Vehicle_Class","Fuel Type":"Fuel_Type","Engine Size(L)":"Engine_Size",
      "Fuel Consumption City (L/100 km)":"Fuel_Consumption_City","Fuel Consumption Hwy (L/100 km)": "Fuel_Consumption_Hwy",
      "Fuel Consumption Comb (L/100 km)": "Fuel_Consumption_Comb",
      "Fuel Consumption Comb (mpg)":"Fuel_Consumption_Comb_","CO2 Emissions(g/km)":"CO2_Emissions"})


# Split data set between target variable and features
X = df_1.copy()
y = X['CO2_Emissions']
X.drop(['CO2_Emissions'], axis=1, inplace=True)

st.title("Numerical Variables Visualization")

st.markdown("""
""")

# vis chart
def plot_num(data, var):
    plt.rcParams['figure.figsize']=(15,5)
    fig=plt.figure()
    plt.subplot(1,3,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('{} histogram'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)


    plt.subplot(1,3,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('{} boxplot'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)


    plt.subplot(1,3,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('{} Q-Q plot'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)

    st.pyplot(fig)

# vis chart
def plot_scatterplot(data,var):
    plt.rcParams['figure.figsize']=(10,5)
    fig=plt.figure()
    sns.scatterplot(data=num2, x=var, y='CO2_Emissions')
    plt.suptitle('CO2_Emissions Distribution per {}'.format(var),fontsize=10)
    plt.xlabel('{}'.format(var), fontsize=15)
    plt.ylabel('CO2_Emissions', fontsize=15)
    plt.yticks(rotation=0,fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    st.pyplot(fig)

# Select numerical columns
numerical_cols = [var for var in X.columns if X[var].dtype in ['int64','float64']]
# Subset with numerical features
num = X[numerical_cols]

plot_num(num, var='Engine_Size')
plot_num(num, var='Cylinders')
plot_num(num, var='Fuel_Consumption_City')
plot_num(num, var='Fuel_Consumption_Hwy')
plot_num(num, var='Fuel_Consumption_Comb')
plot_num(num, var='Fuel_Consumption_Comb_')
num2= pd.concat([y,num], axis=1)
plot_scatterplot(num2, var='Engine_Size')
plot_scatterplot(num2, var='Cylinders')
plot_scatterplot(num2, var='Fuel_Consumption_City')
plot_scatterplot(num2, var='Fuel_Consumption_Hwy')
plot_scatterplot(num2, var='Fuel_Consumption_Comb')
plot_scatterplot(num2, var='Fuel_Consumption_Comb_')




