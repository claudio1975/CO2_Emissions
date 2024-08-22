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
import pickle
import joblib

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Spot Check Models")

st.markdown("""

""")

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
    plt.figure(figsize=(14, 10))
    fig2=plt.figure()
    sns.barplot(x='Bin', y=metric.capitalize(), hue='Model', data=binned_metrics_df, palette=["green", "blue", "orange", "red"])
    plt.axhline(y=0.9 if metric == 'coverage' else 0.45, color='gray', linestyle='--')
    plt.title(f'Binned {metric.capitalize()} for Different Methods')
    plt.xlabel('Bins')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1 if metric == 'coverage' else None)  # Coverage should be between 0 and 1
    plt.legend(title='Method', fontsize=7, title_fontsize=7)
    st.pyplot(fig2)


st.header("Modelling Results")

alpha=0.1

# load datasets
X_test=pd.read_csv('./data/X_test.csv')
X_train=pd.read_csv('./data/X_train.csv')
y_train=pd.read_csv('./data/y_train.csv').squeeze()
y_test=pd.read_csv('./data/y_test.csv').squeeze()


# load the models
LGBM_cqr = pickle.load(open('./data/LGBM_cqr_model.sav', 'rb'))
LGBM_naive = pickle.load(open('./data/LGBM_naive_model.sav', 'rb'))
LGBM_jacknife = pickle.load(open('./data/LGBM_jacknife_model.sav', 'rb'))
LGBM_jacknife_plus = pickle.load(open('./data/LGBM_jacknife_plus_model.sav', 'rb'))
#LGBM_ = pickle.load(open('./data/LGBM_explainer_model.sav', 'rb'))


QR_cqr = pickle.load(open('./data/QR_cqr_model.sav', 'rb'))
QR_naive = pickle.load(open('./data/QR_naive_model.sav', 'rb'))
QR_jacknife = pickle.load(open('./data/QR_jacknife_model.sav', 'rb'))
QR_jacknife_plus = pickle.load(open('./data/QR_jacknife_plus_model.sav', 'rb'))
QR_ = pickle.load(open('./data/QR_explainer_model.sav', 'rb'))


# Prediction

LGBM_cqr_results, LGBM_cqr_predictions_df = calculate_predictions_and_scores(LGBM_cqr,X_test,"QRegressor", alpha)
LGBM_naive_results, LGBM_naive_predictions_df = calculate_predictions_and_scores(LGBM_naive,X_test,"Regressor",alpha)
LGBM_jacknife_results, LGBM_jacknife_predictions_df = calculate_predictions_and_scores(LGBM_jacknife,X_test,"Regressor",alpha)
LGBM_jacknife_plus_results, LGBM_jacknife_plus_predictions_df = calculate_predictions_and_scores(LGBM_jacknife_plus,X_test,"Regressor",alpha)

QR_cqr_results, QR_cqr_predictions_df = calculate_predictions_and_scores(QR_cqr,X_test,"QRegressor", alpha)
QR_naive_results, QR_naive_predictions_df = calculate_predictions_and_scores(QR_naive,X_test,"Regressor",alpha)
QR_jacknife_results, QR_jacknife_predictions_df = calculate_predictions_and_scores(QR_jacknife,X_test,"Regressor",alpha)
QR_jacknife_plus_results, QR_jacknife_plus_predictions_df = calculate_predictions_and_scores(QR_jacknife_plus,X_test,"Regressor",alpha)

#st.subheader("LGBM")

def print_with_title(df, title):
    st.write(f"{title}\n")
    st.write(df)

print_with_title(pd.DataFrame([LGBM_cqr_results]), "LGBM_cqr_results")
print_with_title(pd.DataFrame([LGBM_naive_results]), "LGBM_naive_results")
print_with_title(pd.DataFrame([LGBM_jacknife_results]), "LGBM_jacknife_results")
print_with_title(pd.DataFrame([LGBM_jacknife_plus_results]), "LGBM_jacknife_plus_results")

#st.subheader("QR")

print_with_title(pd.DataFrame([QR_cqr_results]), "QR_cqr_results")
print_with_title(pd.DataFrame([QR_naive_results]), "QR_naive_results")
print_with_title(pd.DataFrame([QR_jacknife_results]), "QR_jacknife_results")
print_with_title(pd.DataFrame([QR_jacknife_plus_results]), "QR_jacknife_plus_results")

st.subheader("Visualization Results")

# Prepare model results and predictions for plotting
LGBM_models_predictions = [
    ("LGBM_CQR", LGBM_cqr_predictions_df),
    ("LGBM_Naive", LGBM_naive_predictions_df),
    ("LGBM_Jackknife", LGBM_jacknife_predictions_df),
    ("LGBM_Jackknife+", LGBM_jacknife_plus_predictions_df)
]

# Prepare model results and predictions for plotting
QR_models_predictions = [
    ("QR_CQR", QR_cqr_predictions_df),
    ("QR_Naive", QR_naive_predictions_df),
    ("QR_Jackknife", QR_jacknife_predictions_df),
    ("QR_Jackknife+", QR_jacknife_plus_predictions_df)
]

#st.subheader("LGBM")

# Plot LGBM model errors
st.write("LGBM model errors")
plot_error(LGBM_models_predictions, calculate_errors)

# Plot LGBM model predictions
st.write("LGBM model predictions")
plot_prediction(LGBM_models_predictions)

binned_coverage_df = main_binned_metric(LGBM_cqr_predictions_df, LGBM_naive_predictions_df, LGBM_jacknife_predictions_df, LGBM_jacknife_plus_predictions_df, metric='coverage')
st.write("LGBM binned coverage")
plot_binned_metric('coverage', binned_coverage_df)

binned_width_df = main_binned_metric(LGBM_cqr_predictions_df, LGBM_naive_predictions_df, LGBM_jacknife_predictions_df, LGBM_jacknife_plus_predictions_df, metric='width')
st.write("LGBM binned width")
plot_binned_metric('width', binned_width_df)

#st.subheader("QR")

# Plot QR model errors
st.write("QR model errors")
plot_error(QR_models_predictions, calculate_errors)

# Plot LR model predictions
st.write("QR model predictions")
plot_prediction(QR_models_predictions)

binned_coverage_df = main_binned_metric(QR_cqr_predictions_df, QR_naive_predictions_df, QR_jacknife_predictions_df, QR_jacknife_plus_predictions_df, metric='coverage')
st.write("QR binned coverage")
plot_binned_metric('coverage', binned_coverage_df)

binned_width_df = main_binned_metric(QR_cqr_predictions_df, QR_naive_predictions_df, QR_jacknife_predictions_df, QR_jacknife_plus_predictions_df, metric='width')
st.write("QR binned width")
plot_binned_metric('width', binned_width_df)

st.subheader("Feature Importance")

# Global SHAP on LGBM
#fig=plt.figure()
#LGBM_explainer = shap.TreeExplainer(LGBM_)
#LGBM_shap_values = LGBM_explainer.shap_values(X_test)
#plt.rcParams['figure.figsize'] = (5,5)
#st.write("LGBM SHAP FEATURES IMPORTANCE ON CO2 EMISSIONS")
#shap.summary_plot(LGBM_shap_values, features=X_test, feature_names=X_test.columns, plot_type='bar', show=False)
#st.pyplot(fig)

# Global SHAP on QR
fig=plt.figure()
masker = shap.maskers.Independent(X_train)
QR_explainer = shap.LinearExplainer(QR_, masker=masker)
QR_shap_values = QR_explainer.shap_values(X_test)
plt.rcParams['figure.figsize'] = (5,5)
st.write("QR SHAP FEATURES IMPORTANCE ON CO2 EMISSIONS")
shap.summary_plot(QR_shap_values, features=X_test, feature_names=X_test.columns, plot_type='bar', show=False)
st.pyplot(fig)
