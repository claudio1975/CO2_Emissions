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

st.title("Categorical Variables Visualization")

st.markdown("""
""")

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [var for var in X.columns if
                    X[var].nunique() <= 50 and 
                    X[var].dtype == "object"]

# Subset with categorical features
cat = X[categorical_cols]

# vis chart
def plot_cat(data, var):
    plt.rcParams['figure.figsize']=(15,5)
    fig=plt.figure()
    sns.countplot(x=data[var], data=data).set_title("Barplot {} Variable Distribution".format(var))
    plt.yticks(rotation=0, fontsize=5)
    plt.xticks(rotation=90, fontsize=10)
    st.pyplot(fig)

# Bivariate analysis with box-plots
def plot_boxplot2(data, var):
    plt.rcParams['figure.figsize']=(20,10)
    fig=plt.figure()
    sns.boxplot(x=data[var], y='CO2_Emissions', linewidth=2, palette="Set1", data=data)
    plt.suptitle('CO2_Emissions Distribution per {}'.format(var),fontsize=10)
    plt.xlabel('{}'.format(var), fontsize=15)
    plt.ylabel('CO2_Emissions', fontsize=15)
    plt.yticks(rotation=0,fontsize=15)
    plt.xticks(rotation=90, fontsize=15)
    st.pyplot(fig)

plot_cat(cat, var = 'Make')
plot_cat(cat, var='Vehicle_Class')
plot_cat(cat, var='Transmission')
plot_cat(cat,var='Fuel_Type')
cat2=pd.concat([y,cat], axis=1)
plot_boxplot2(cat2, var='Make')
plot_boxplot2(cat2, var='Vehicle_Class')
plot_boxplot2(cat2, var='Transmission')
plot_boxplot2(cat2, var='Fuel_Type')


