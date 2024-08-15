import streamlit as st

st.title("Introduction")

st.markdown("""
The goal is to predict CO2 emissions by vehicles using Conformal Prediction. 
For this purpose are employed two models: LightGBM and Quantile Regressor.

Four conformal prediction methods are compared for each model: 
            
-Conformalized Quantile Regression;
            
-The Naive method;
            
-The Jackknife method;    
                    
-The Jackknife+ method.

This dataset captures the details of how CO2 emissions by a vehicle can vary with the different features. 
The dataset has been taken from Canada Government official open data website. This is a compiled version. 
This contains data over a period of 7 years.
There are total 7385 rows and 12 columns. 
            
### Data Description

#### Variables
            
**Model**
            
4WD/4X4 = Four-wheel drive
            
AWD = All-wheel drive
            
FFV = Flexible-fuel vehicle
            
SWB = Short wheelbase
            
LWB = Long wheelbase
            
EWB = Extended wheelbase

**Transmission**
            
A = Automatic
            
AM = Automated manual
            
AS = Automatic with select shift
            
AV = Continuously variable
            
M = Manual
            
3 - 10 = Number of gears

**Fuel type**
            
X = Regular gasoline
            
Z = Premium gasoline
            
D = Diesel
            
E = Ethanol (E85)
            
N = Natural gas

**Fuel Consumption**
            
City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km) 
The combined rating is shown in L/100 km and in miles per gallon (mpg)

**CO2 Emissions**
            
The tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving


#### Evaluation Metric

The evaluation metric used for this data set is the RMSE score

* Data set source: https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles

            
* ACKNOWLEDGEMENTS
            
The data has been taken and compiled from the Canada Government: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6

""")
