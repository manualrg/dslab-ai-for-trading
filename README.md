# dslab-ai-for-trading
Advanced Analytics and Machine Learning applied to Financial Markets and Quantitative Trading on Python

## Introduction
This project is a compilation on of several application of ML models to trading.

## Setup
Two environments are created to run, the first one (quant-ai4trading) is focused to run Zipline and Alphalens (from Quantopian) and the second one (quant-tsa) has library to perform quantitative and time series analysis

Run `conda env create -f environment-quantai4trading.yaml` or `conda env create -fenvironment-quanttsa.yaml`

## Index
Project structure is based con Data Science CookieCutter's template: https://drivendata.github.io/cookiecutter-data-science/

In notebooks, the following folders can be found:

### 00_load_data
Load data from external soures, like Quandl or SimFin
Run on any environmnet

### 01_alpha_factors
Alpha factors are features or variables that may forecast future returns in a broad set of market assets.
Use Zipline to load eod data and compute alpha factors using Zipline. Perform alpha factor evaluation with Alphalens.
Run on quant-ai4trading environment

### 02_risk_factors
On the other hand, a risk factor may explain common variance or volatility shared among a wide set of market assets
Use Zipline to load eod data and compute risk factors models
Run on quant-ai4trading environment

### 03_ml_models
Apply ML models to combine several alpha factors and other quant features in a ml-alpha-factor
Train and evaluate and bencharmk several alpha models
Run on quant-tsa environment

### 04_opt_portfolio
Perform portfolio optimization based on alpha and risk model predictions. Also backtest portfolio building strategy and evaluate portfolio performance
Run on quant-tsa environment

