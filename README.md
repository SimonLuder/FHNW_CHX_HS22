FHNW_chx_HS22
==============================

This repository is used to develop and structure code for the Challenge-X, which is being carry out as a group project in the Data Science FHNW course during the autumn semester of 2022.


Team Members
------------

- Cédric Künzi 

- Simon Luder


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    |    └── archive       <- Old notebooks
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python modules
        │
        └── Dashboard      <- Code to cretae the dashboard, with the data and the model

--------

Data:

After downloading the code, a folder data/raw in the root project has to be made. The data has to be loaded in raw. The data can be found in the Teams project.

Exploration:

The notebooks for the data exploration can be found in the notebooks folder. 
- eda_timeseries.ipynb: exploration of the time series data
- Forecasting_CNN.ipynb: different CNN models for forecasting the number of available parking spaces in a parking
- Forecasting_different_parkings.ipynb: forecasting for different parkings using CNN models
- Forecasting_with_Deep_Learning.ipynb: forecasting with LSTM models
- prep_live_data.ipynb: cleaning and forecasting with LSTM on live data
- live_data_collector.ipynb: code for collecting live data
- eda_corona_historic.ipynb: Shows the influence of the corona pandemic on the number of available parking spaces
- Eda_statistical_forecasting.ipynb: forecasting using statistical methods

- model_architectures.py: contains the different model architectures used in the notebooks
- helper.py: contains helper functions used in the notebooks (for data cleaning etc.)
- dashboard_data_api.py: contains the functions to get the live data for the dashboard
- live_data_collector.py: contains the functions to collect the live data
- clean_data.py: first function to clean the data

The most important files are the eda_timeseries.ipynb for the data exploration, the Eda_statistical_forecasting.ipynb for the exploration of statistical methods for forecasting, the Forecasting_with_Deep_Learning.ipynb and Foreasting_CNN.ipynb for the exploration of deep learning methods for forecasting and the live_data_collector.ipynb for the live data collection.

Dashbaord:

In the src/Dashboard folder is the code for the dashboard.
- app.py: contains the code for the dashboard
- function_class.py: contains the functions for the dashboard (model architecture, distance calculation etc.)
- test_class.py: contains the test functions for the dashboard
- test_data.csv/test_data1.txt: contains the test data for the dashboard (fake data)
- models folder: contains the models for the dashboard

The dashboard can be started in the command shell. For this you have to be in the folder for this project and then type in the command shell: 'python -m src.Dashboard.app'

To find the closest parking, first click on the dashboard where you want to go. Then click where you are. The distance to the closest parking will be shown in the dashboard.