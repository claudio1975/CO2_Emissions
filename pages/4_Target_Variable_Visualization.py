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

df = pd.read_csv('./CO2_Emissions_Canada.csv')

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

st.title("Target Variable Visualization")


st.markdown("""

""")


# Vis Chart
def plot_target(data, var):
    plt.rcParams['figure.figsize']=(15,5)
    fig = plt.figure()
    plt.suptitle('CO2 Emissions Data Analysis',fontsize=15)
    plt.subplot(1,3,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('{} histogram'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)

    plt.subplot(1,3,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('{} boxplot'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)


    plt.subplot(1,3,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('{} Q-Q plot'.format(var))
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)

    st.pyplot(fig)

plot_target(df_1, var='CO2_Emissions')



