import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
import matplotlib
matplotlib.use('TkAgg')  # You can also try 'Qt5Agg' or 'Agg' if this doesn't work
import matplotlib.pyplot as plt

def read_csv(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path)
    return data


# Data Summary Function
def Data_Summary(data):
    print("Data Summary for file: ", file)
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    # print(data.iloc[:, -1].describe())
    # # Visualize the last column
    # temp_data = data.drop(data.index[0])
    # plt.figure(figsize=(7, 6))
    # plt.bar(temp_data.iloc[:, -1].unique(), temp_data.iloc[:, -1].value_counts())
    # plt.show(block=True)

    print("---------------------------------------------------------------------------------")


def User_Features_Preparation(data):
    for column in data.columns:
        # fill first_prediction with the average of the column
        if column == 'first_prediction':
            data[column].fillna(data[column].mean(), inplace=True)

        # check if column contains null values
        elif data[column].isnull().sum() > 0:
            # fill the null values with the mean of the column
            data[column].fillna(0, inplace=True)

    data = Category(data)
    return data


def Train_Data_Preparation(data):
    for column in data.columns:
        # check if column contains null values
        if data[column].isnull().sum() > 0:
            # fill the null values with the mean of the column
            data[column].fillna(0, inplace=True)
    data = Category(data)

    return data


def Category(data):
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].dtype == 'bool':
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

    return data


# Megre the two dataframesUsing the first column as the key
def Merge_Data(user_features, user_train, output_file):
    merged_data = pd.merge(user_features, user_train, on='ID')
    # Data_Summary(merged_data)
    merged_data = pd.merge(merged_data, output_file, on='ID')

    return merged_data


def Merge_Test_Data(user_features, user_train):
    merged_data = pd.merge(user_features, user_train, on='ID')

    return merged_data


def Train_Test_Data(merged_data):
    # split the data into train and test
    X = merged_data.drop('TARGET', axis=1)
    X = X.drop('ID', axis=1)
    y = merged_data['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print("Train Data")
    # print(X_train.head())
    # print(y_train.head())
    # print("Test Data")
    # print(X_test.head())
    # print(y_test.head())

    return X_train, X_test, y_train, y_test


def Machine_Learning(X_train, X_test, y_train, y_test):
    # Define the models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
    }

    # Dictionary to store trained models and their scores
    trained_models = {}
    scores = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        trained_models[name] = model
        scores[name] = rmse
        print(f"{name} - RMSE: {rmse:.4f}")

    return trained_models, scores


def Test_Data_Preparation(data):
    for column in data.columns:
        # check if column contains null values
        if data[column].isnull().sum() > 0:
            # fill the null values with the mean of the column
            data[column].fillna(0, inplace=True)
    data = Category(data)

    return data


def compare_first_prediction_vs_target(data):
    # devide teh data into two parts at 10% and 90%
    # data = data.iloc[:int(len(data) * 0.1)]
    """
    Compares the contribution of 'android' and 'ios' platforms to the TARGET column.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the 'platform' and 'TARGET' columns.
    """
    # Check if required columns are in the data
    if 'platform' not in data.columns or 'TARGET' not in data.columns:
        print("Columns 'platform' and 'TARGET' are required in the data.")
        return

    # print the different types of platforms and their count
    print(data['platform'].unique())

    # Group by platform and calculate mean TARGET value
    platform_contribution = data.groupby('platform')['TARGET'].mean().reset_index()

    # Display the mean TARGET values for each platform
    print(platform_contribution)

    # add the names android and IOS to the platform column
    platform_contribution['platform'] = platform_contribution['platform'].replace({0: 'Android', 1: 'iOS'})

    # Plotting the comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(platform_contribution['platform'], platform_contribution['TARGET'], color=['blue', 'green'])
    plt.title('Platform Contribution to Target')
    plt.xlabel('Platform')
    plt.ylabel('Average TARGET Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding labels on the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.show()

if __name__ == '__main__':
    # get all the diles in the Data folder
    file_path = 'Data/Train/'
    user_features = pd.DataFrame()
    user_train = pd.DataFrame()

    for file in os.listdir(file_path):
        file = file_path + file

        data = read_csv(file)
        # Data_Summary(data)
        if file == 'Data/Train/user_features_train.csv':
            user_features = User_Features_Preparation(data)
            # Data_Summary(user_features)
        else:
            user_train = Train_Data_Preparation(data)
            # Data_Summary(user_train)

    output = read_csv('Data/targets_train.csv')
    # Data_Summary(output)
    merged_data = Merge_Data(user_features, user_train, output)
    # Data_Summary(merged_data)
    # print(merged_data.head(15))

    # merged_data.drop('first_prediction', axis=1, inplace=True)
    # merged_data.drop('first_open_date', axis=1, inplace=True)
    merged_data.drop('first_open_timestamp', axis=1, inplace=True)
    merged_data.drop('local_first_open_timestamp', axis=1, inplace=True)

    compare_first_prediction_vs_target(merged_data)

    # X_train, X_test, y_train, y_test = Train_Test_Data(merged_data)
    # models, scores = Machine_Learning(X_train, X_test, y_train, y_test)
#
    # print("Test Data")
    # file_path = 'Data/Test/'
    # user_test = pd.DataFrame()
    # user_features = pd.DataFrame()
    # for file in os.listdir(file_path):
    #     file = file_path + file
#
    #     data = read_csv(file)
    #     # Data_Summary(data)
    #     if file == 'Data/Test/user_features_test.csv':
    #         user_features = User_Features_Preparation(data)
    #         # Data_Summary(user_features)
    #     else:
    #         user_test = Test_Data_Preparation(data)
    #         # Data_Summary(user_test)
#
    # merged_test_data = Merge_Test_Data(user_features, user_test)
#
    # merged_test_data.drop('first_prediction', axis=1, inplace=True)
    # # merged_test_data.drop('first_open_date', axis=1, inplace=True)
    # merged_test_data.drop('first_open_timestamp', axis=1, inplace=True)
    # merged_test_data.drop('local_first_open_timestamp', axis=1, inplace=True)
    # ID_column = merged_test_data['ID']
    # merged_test_data.drop('ID', axis=1, inplace=True)
#
    # # print(merged_test_data.head(15))
#
    # # Make predictions on the test data
    # y_pred = models['Ridge Regression'].predict(merged_test_data)
    # print(y_pred)
    # # Merge ID and y_pred and Save the predictions to a CSV file
    # predictions = pd.DataFrame({'ID': ID_column, 'TARGET': y_pred})
    # predictions.to_csv('Data/Ridge Regression predictions.csv', index=False)
#
    # print("Predictions saved to 'Data/predictions.csv'")
#
    # y_pred = models['Linear Regression'].predict(merged_test_data)
    # print(y_pred)
    # # Merge ID and y_pred and Save the predictions to a CSV file
    # predictions = pd.DataFrame({'ID': ID_column, 'TARGET': y_pred})
    # predictions.to_csv('Data/Linear Regression predictions.csv', index=False)
#
    # print("Done")
