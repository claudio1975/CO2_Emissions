import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro
import optuna
from lightgbm import LGBMRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score
import shap
import streamlit as st

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Spot Check Models")

st.markdown("""

""")

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

# Functions
def calculate_predictions_and_scores(Model,X_test,regressor_type, alpha):
    # Make predictions on the test data
    if regressor_type == "QRegressor":
        y_pred, y_pis = Model.predict(X_test)
    else:
        y_pred, y_pis = Model.predict(X_test, alpha=alpha)

    # Store predictions in a dataframe
    predictions = y_test.to_frame()
    predictions.columns = ['y_true']
    predictions["point_prediction"] = y_pred
    predictions["lower"] = y_pis.reshape(-1, 2)[:, 0]
    predictions["upper"] = y_pis.reshape(-1, 2)[:, 1]

    # Calculate the coverage and width of the prediction intervals
    coverage = regression_coverage_score(
        y_test,
        y_pis[:, 0, 0],
        y_pis[:, 1, 0]
    )
    width = regression_mean_width_score(
        y_pis[:, 0, 0],
        y_pis[:, 1, 0]
    )

    # Calculate RMSE for lower, point, and upper predictions
    score_lower = np.sqrt(mean_squared_error(predictions[['y_true']], predictions[['lower']]))
    score_median = np.sqrt(mean_squared_error(predictions[['y_true']], predictions[['point_prediction']]))
    score_upper = np.sqrt(mean_squared_error(predictions[['y_true']], predictions[['upper']]))

    results = {
        "coverage": coverage,
        "width": width,
        "score_lower": score_lower,
        "score_median": score_median,
        "score_upper": score_upper
    }

    return results, predictions  # Return results and predictions separately

# Plot Errors
# Calculate errors for each model
def calculate_errors(predictions):
    error_lower = predictions["point_prediction"] - predictions["lower"]
    error_upper = predictions["upper"] - predictions["point_prediction"]

    # Ensure no negative values in yerr. If negative, make them zero.
    error_lower = np.maximum(error_lower, 0)
    error_upper = np.maximum(error_upper, 0)

    # Combine lower and upper errors into one array
    error = [error_lower, error_upper]
    return error


# Plotting in a 2x2 grid
def plot_error(models_predictions, calculate_errors):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (model_name, predictions) in zip(axes, models_predictions):
        error = calculate_errors(predictions)

        ax.errorbar(predictions["y_true"], predictions["point_prediction"],
                yerr=error,
                ecolor='gray', linestyle='', marker="o", capsize=8)

        ax.axline([0, 0], [1, 1], color="red", linestyle='--', lw=3, zorder=3)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_xlabel('True CO2 Emissions')
        ax.set_ylabel('Predicted CO2 Emissions')
        ax.set_title(f'{model_name} Method')

    plt.tight_layout()
    st.pyplot(fig)

# Plot Predictions
# Plotting in a 2x2 grid
def plot_prediction(models_predictions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (model_name, predictions) in zip(axes, models_predictions):
        # Re-sort for plot
        sorted_predictions = predictions.sort_values(by=['y_true']).reset_index(drop=True)

        ax.plot(sorted_predictions["y_true"], 'o', markersize=5, color='blue', label="y_true")
        ax.plot(sorted_predictions["point_prediction"], 'o', markersize=3, color='red',label="y_prediction")
        ax.fill_between(np.arange(len(sorted_predictions)),
                    sorted_predictions["lower"],
                    sorted_predictions["upper"],
                    alpha=0.8, color="green", label="prediction interval")

        ax.set_xticks([])
        ax.set_xlim([0, len(sorted_predictions)])
        ax.set_ylabel("True value")
        ax.set_title(f'{model_name} Method')
        ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

# General function to calculate binned coverage or width
def calculate_binned_metric(predictions, bin_edges, metric='coverage'):
    predictions['bin'] = np.digitize(predictions['point_prediction'], bin_edges, right=True)
    bins = predictions['bin'].unique()
    binned_metric = {}
    for bin in bins:
        bin_data = predictions[predictions['bin'] == bin]
        if metric == 'coverage':
            metric_value = ((bin_data['y_true'] >= bin_data['lower']) & (bin_data['y_true'] <= bin_data['upper'])).mean()
        elif metric == 'width':
            metric_value = (bin_data['upper'] - bin_data['lower']).mean()
        binned_metric[bin] = metric_value
    return binned_metric

# Function to get combined bin edges
def get_bin_edges(*prediction_frames, n_bins=10):
    all_point_predictions = pd.concat([df['point_prediction'] for df in prediction_frames])
    bin_edges = np.linspace(all_point_predictions.min(), all_point_predictions.max(), n_bins)
    return bin_edges

# General function to combine binned metrics
def combine_binned_metrics(binned_metrics_dict, metric='Coverage'):
    data = []
    for model_name, metrics in binned_metrics_dict.items():
        for bin, value in metrics.items():
            data.append([bin, model_name, value])
    return pd.DataFrame(data, columns=['Bin', 'Model', metric])

# Main functions to calculate and combine binned coverage and width
def main_binned_metric(cqr_predictions_df, naive_predictions_df, jacknife_predictions_df, jacknife_plus_predictions_df, metric='coverage'):
    bin_edges = get_bin_edges(cqr_predictions_df, naive_predictions_df, jacknife_predictions_df, jacknife_plus_predictions_df)
    metrics_funcs = {
        'coverage': calculate_binned_metric,
        'width': calculate_binned_metric
    }

    binned_metrics = {
        "CQR": metrics_funcs[metric](cqr_predictions_df, bin_edges, metric),
        "Naive": metrics_funcs[metric](naive_predictions_df, bin_edges, metric),
        "Jacknife": metrics_funcs[metric](jacknife_predictions_df, bin_edges, metric),
        "Jacknife+": metrics_funcs[metric](jacknife_plus_predictions_df, bin_edges, metric)
    }

    binned_metrics_df = combine_binned_metrics(binned_metrics, metric.capitalize())

    return binned_metrics_df

# Plotting binned metric
def plot_binned_metric(metric, binned_metrics_df):
    plt.figure(figsize=(8, 5))
    fig=plt.figure()
    sns.barplot(x='Bin', y=metric.capitalize(), hue='Model', data=binned_metrics_df, palette=["green", "blue", "orange", "red"])
    plt.axhline(y=0.9 if metric == 'coverage' else 0.45, color='gray', linestyle='--')
    plt.title(f'Binned {metric.capitalize()} for Different Methods')
    plt.xlabel('Bins')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1 if metric == 'coverage' else None)  # Coverage should be between 0 and 1
    plt.legend(title='Method', fontsize=7, title_fontsize=7)
    st.pyplot(fig)


# Select numerical columns
numerical_cols = [var for var in X.columns if X[var].dtype in ['int64','float64']]

# Subset with numerical features
num = X[numerical_cols]
num2= pd.concat([y,num], axis=1)


# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() <= 50 and
                    X[cname].dtype == "object"]

cat=X[categorical_cols]
cat2=pd.concat([y,cat], axis=1)



# IQR
# Calculate the upper and lower limits
Q1=df_1['CO2_Emissions'].quantile(0.25)
Q3=df_1['CO2_Emissions'].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

# Create arrays of Boolean values indicating the outlier rows
upper_array = df_1[df_1['CO2_Emissions'] > upper].index
lower_array = df_1[df_1['CO2_Emissions'] < lower].index

# Removing the outliers by the target
df_2=df_1.drop(index=upper_array, axis=1)
df_2=df_2.drop(index=lower_array, axis=1)


# Split data set between target variable and features
X_ = df_2.copy()
y_ = X_['CO2_Emissions']
X_.drop(['CO2_Emissions'], axis=1, inplace=True)
num_=X_[numerical_cols]
cat_=X_[categorical_cols]

# cap residual outliers on features
i = 'Engine_Size'
q75, q25 = np.percentile(num_[i].dropna(), [75 ,25])
iqr = q75 - q25
min_val = q25 - (iqr*1.5)
max_val = q75 + (iqr*1.5)

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
num_out = num_.copy()

# Use .loc to set the values within the IQR range
num_out.loc[num_out[i] < min_val, i] = min_val
num_out.loc[num_out[i] > max_val, i] = max_val

# cap residual outliers on features
i = 'Cylinders'
q75, q25 = np.percentile(num_[i].dropna(), [75 ,25])
iqr = q75 - q25
min_val = q25 - (iqr*1.5)
max_val = q75 + (iqr*1.5)

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
num_out = num_.copy()

# Use .loc to set the values within the IQR range
num_out.loc[num_out[i] < min_val, i] = min_val
num_out.loc[num_out[i] > max_val, i] = max_val

# cap residual outliers on features
i = 'Fuel_Consumption_City'
q75, q25 = np.percentile(num_[i].dropna(), [75 ,25])
iqr = q75 - q25
min_val = q25 - (iqr*1.5)
max_val = q75 + (iqr*1.5)

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
num_out = num_.copy()

# Use .loc to set the values within the IQR range
num_out.loc[num_out[i] < min_val, i] = min_val
num_out.loc[num_out[i] > max_val, i] = max_val

# cap residual outliers on features
i = 'Fuel_Consumption_Hwy'
q75, q25 = np.percentile(num_[i].dropna(), [75 ,25])
iqr = q75 - q25
min_val = q25 - (iqr*1.5)
max_val = q75 + (iqr*1.5)

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
num_out = num_.copy()

# Use .loc to set the values within the IQR range
num_out.loc[num_out[i] < min_val, i] = min_val
num_out.loc[num_out[i] > max_val, i] = max_val

# cap residual outliers on features
i='Fuel_Consumption_Comb'
q75,q25=np.percentile(num_[i].dropna(), [75,25])
iqr=q75-q25
min_val=q25-(iqr*1.5)
max_val=q75+(iqr*1.5)

# create a copy of the dataframe to avoid Setting With Copy Warning
num_out=num_.copy()

# use .loc to set the values within the IQR range
num_out.loc[num_out[i]<min_val,i]=min_val
num_out.loc[num_out[i]>max_val,i]=max_val

# cap residual outliers on features
i = 'Fuel_Consumption_Comb_'
q75, q25 = np.percentile(num_[i].dropna(), [75 ,25])
iqr = q75 - q25
min_val = q25 - (iqr*1.5)
max_val = q75 + (iqr*1.5)

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
num_out = num_.copy()

# Use .loc to set the values within the IQR range
num_out.loc[num_out[i] < min_val, i] = min_val
num_out.loc[num_out[i] > max_val, i] = max_val


# Encoding Categorical Variables

cat2_=pd.concat([cat_,y_], axis=1)

# calculate the mean target value per category for each feature and capture the result in a dictionary
MAKE_LABELS = cat2_.groupby(['Make'])['CO2_Emissions'].mean().to_dict()
VEHICLE_CLASS_LABELS = cat2_.groupby(['Vehicle_Class'])['CO2_Emissions'].mean().to_dict()
TRASMISSION_LABELS = cat2_.groupby(['Transmission'])['CO2_Emissions'].mean().to_dict()
FUEL_TYPE_LABELS = cat2_.groupby(['Fuel_Type'])['CO2_Emissions'].mean().to_dict()

# replace for each feature the labels with the mean target values
cat2_['Make'] = cat2_['Make'].map(MAKE_LABELS)
cat2_['Vehicle_Class'] = cat2_['Vehicle_Class'].map(VEHICLE_CLASS_LABELS)
cat2_['Transmission'] = cat2_['Transmission'].map(TRASMISSION_LABELS)
cat2_['Fuel_Type'] = cat2_['Fuel_Type'].map(FUEL_TYPE_LABELS)

# Look at the new subset
target_cat = cat2_.drop(['CO2_Emissions'], axis=1)
X_all=pd.concat([target_cat, num_out], axis=1)

# Zero Variance Predictors

# Find features with variance equal zero or lower than 0.05
to_drop = [col for col in X_all.columns if np.var(X_all[col]) ==0]
# Drop features
X_all_v = X_all.drop(X_all[to_drop], axis=1)


# Correlated Predictors

# Correlation heatmap
corr_matrix = X_all.corr(method='spearman')
# Select correlated features and removed it
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find index of feature columns with correlation greater than 0.75
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.75)]
# Drop features
X_all_f = X_all.drop(X_all[to_drop], axis=1)


# Split Data Set

# Train/calibration/test split
X_train_cal, X_test, y_train_cal, y_test = train_test_split(X_all_f, y_, test_size=0.1)
X_train, X_cal, y_train, y_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.1)

scaling=MinMaxScaler()
X_all_sc=pd.DataFrame(scaling.fit_transform(X_all_f),columns=['Make', 'Vehicle_Class', 'Transmission', 'Fuel_Type', 'Engine_Size'])

# Train/calibration/test split
X_train_cal_sc, X_test_sc, y_train_cal_sc, y_test_sc = train_test_split(X_all_sc, y_, test_size=0.1)
X_train_sc, X_cal_sc, y_train_sc, y_cal_sc = train_test_split(X_train_cal_sc, y_train_cal_sc, test_size=0.1)

st.header("Modelling Results")

alpha=0.1

# LGBM Optimized Estimator

LGBM = LGBMRegressor(
        objective='quantile',
        alpha=0.5,
        n_estimators= 604,
        learning_rate= 0.028182937892429497,
        max_depth= 7,
        min_child_samples=39,
        num_leaves= 130,
        n_jobs=-1,
        random_state=0)

# LR Optimized Estimator

LR=QuantileRegressor(quantile=0.5,alpha=0.29643031153893773, solver='highs')

# Fitting models

LGBM_cqr = MapieQuantileRegressor(estimator=LGBM, cv="split", alpha=alpha, method= "quantile")
LGBM_cqr.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal, random_state=0)

np.random.seed(0)
LR_cqr = MapieQuantileRegressor(estimator=LR, cv="split", alpha=alpha, method= "quantile")
LR_cqr.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal, random_state=0)


np.random.seed(0)
LGBM_naive = MapieRegressor(estimator=LGBM, method= "naive")
LGBM_naive.fit(X_train, y_train)

np.random.seed(0)
LR_naive = MapieRegressor(estimator=LR,method= "naive")
LR_naive.fit(X_train, y_train)


np.random.seed(0)
LGBM_jacknife = MapieRegressor(estimator=LGBM, method= "base", cv=5)
LGBM_jacknife.fit(X_train, y_train)

np.random.seed(0)
LR_jacknife = MapieRegressor(estimator=LR,method= "base", cv=5)
LR_jacknife.fit(X_train, y_train)


np.random.seed(0)
LGBM_jacknife_plus = MapieRegressor(estimator=LGBM, method= "plus", cv=5)
LGBM_jacknife_plus.fit(X_train, y_train)

np.random.seed(0)
LR_jacknife_plus = MapieRegressor(estimator=LR,method= "plus", cv=5)
LR_jacknife_plus.fit(X_train, y_train)

# Prediction

LGBM_cqr_results, LGBM_cqr_predictions_df = calculate_predictions_and_scores(LGBM_cqr,X_test,"QRegressor", alpha)
LGBM_naive_results, LGBM_naive_predictions_df = calculate_predictions_and_scores(LGBM_naive,X_test,"Regressor",alpha)
LGBM_jacknife_results, LGBM_jacknife_predictions_df = calculate_predictions_and_scores(LGBM_jacknife,X_test,"Regressor",alpha)
LGBM_jacknife_plus_results, LGBM_jacknife_plus_predictions_df = calculate_predictions_and_scores(LGBM_jacknife_plus,X_test,"Regressor",alpha)

LR_cqr_results, LR_cqr_predictions_df = calculate_predictions_and_scores(LR_cqr,X_test,"QRegressor", alpha)
LR_naive_results, LR_naive_predictions_df = calculate_predictions_and_scores(LR_naive,X_test,"Regressor",alpha)
LR_jacknife_results, LR_jacknife_predictions_df = calculate_predictions_and_scores(LR_jacknife,X_test,"Regressor",alpha)
LR_jacknife_plus_results, LR_jacknife_plus_predictions_df = calculate_predictions_and_scores(LR_jacknife_plus,X_test,"Regressor",alpha)

st.subheader("LGBM")

LGBM_cqr_results
LGBM_naive_results
LGBM_jacknife_results
LGBM_jacknife_plus_results

st.subheader("QR")

LR_naive_results
LR_cqr_results
LR_jacknife_results
LR_jacknife_plus_results

st.subheader("Visualization Results")

# Prepare model results and predictions for plotting
LGBM_models_predictions = [
    ("LGBM_CQR", LGBM_cqr_predictions_df),
    ("LGBM_Naive", LGBM_naive_predictions_df),
    ("LGBM_Jackknife", LGBM_jacknife_predictions_df),
    ("LGBM_Jackknife+", LGBM_jacknife_plus_predictions_df)
]

# Prepare model results and predictions for plotting
LR_models_predictions = [
    ("LR_CQR", LR_cqr_predictions_df),
    ("LR_Naive", LR_naive_predictions_df),
    ("LR_Jackknife", LR_jacknife_predictions_df),
    ("LR_Jackknife+", LR_jacknife_plus_predictions_df)
]

st.subheader("LGBM")

# Plot LGBM model errors
plot_error(LGBM_models_predictions, calculate_errors)

# Plot LGBM model predictions
plot_prediction(LGBM_models_predictions)

binned_coverage_df = main_binned_metric(LGBM_cqr_predictions_df, LGBM_naive_predictions_df, LGBM_jacknife_predictions_df, LGBM_jacknife_plus_predictions_df, metric='coverage')
plot_binned_metric('coverage', binned_coverage_df)

binned_width_df = main_binned_metric(LGBM_cqr_predictions_df, LGBM_naive_predictions_df, LGBM_jacknife_predictions_df, LGBM_jacknife_plus_predictions_df, metric='width')
plot_binned_metric('width', binned_width_df)

st.subheader("QR")

# Plot QR model errors
plot_error(LR_models_predictions, calculate_errors)

# Plot LR model predictions
plot_prediction(LR_models_predictions)

binned_coverage_df = main_binned_metric(LR_cqr_predictions_df, LR_naive_predictions_df, LR_jacknife_predictions_df, LR_jacknife_plus_predictions_df, metric='coverage')
plot_binned_metric('coverage', binned_coverage_df)

binned_width_df = main_binned_metric(LR_cqr_predictions_df, LR_naive_predictions_df, LR_jacknife_predictions_df, LR_jacknife_plus_predictions_df, metric='width')
plot_binned_metric('width', binned_width_df)

st.subheader("Feature Importance")

# Global SHAP on LGBM
fig=plt.figure()
LGBM_ = LGBMRegressor(
        objective='quantile',
        alpha=0.5,
        n_estimators= 604,
        learning_rate= 0.028182937892429497,
        max_depth= 7,
        min_child_samples=39,
        num_leaves= 130,
        n_jobs=-1,
        random_state=0).fit(X_train, y_train)
LGBM_explainer = shap.TreeExplainer(LGBM_)
LGBM_shap_values = LGBM_explainer.shap_values(X_test)
plt.rcParams['figure.figsize'] = (5,5)
st.write("LGBM SHAP FEATURES IMPORTANCE ON CO2 EMISSIONS")
shap.summary_plot(LGBM_shap_values, features=X_test, feature_names=X_all_f.columns, plot_type='bar', show=False)
st.pyplot(fig)

# Global SHAP on QR
fig=plt.figure()
QR_ = QuantileRegressor(quantile=0.5, alpha=0.29643031153893773, solver='highs').fit(X_train, y_train)
masker = shap.maskers.Independent(X_train)
QR_explainer = shap.LinearExplainer(QR_, masker=masker)
QR_shap_values = QR_explainer.shap_values(X_test)
plt.rcParams['figure.figsize'] = (5,5)
st.write("QR SHAP FEATURES IMPORTANCE ON CO2 EMISSIONS")
shap.summary_plot(QR_shap_values, features=X_test, feature_names=X_test.columns, plot_type='bar', show=False)
st.pyplot(fig)