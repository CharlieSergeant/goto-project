from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap


class CompanySize(Enum):
    LARGE = 3
    MID = 2
    SMALL = 1
    NOT_AVAILABLE = 0


class MarketType(Enum):
    NA = 4
    LATAM = 3
    EMEA = 2
    AUS_ROW = 1
    NOT_AVAILABLE = 0


def apply_company_size(row):
    """
    Assign the company size category based on the presence of a specific column.

    Parameters:
        row (pd.Series): A row of the DataFrame.

    Returns:
        int: The numerical value representing the company size category.
    """
    if row['company_size_large_size_business'] == 1:
        return CompanySize.LARGE.value
    elif row['company_size_mid_size_business'] == 1:
        return CompanySize.MID.value
    elif row['company_size_small_business'] == 1:
        return CompanySize.SMALL.value
    elif row['company_size_not_available'] == 1:
        return CompanySize.NOT_AVAILABLE.value
    else:
        return CompanySize.NOT_AVAILABLE.value  # Default to 0 for 'Unknown'


def transform_company_size(df):
    """
    Transform the company size columns into a single 'company_size' column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'company_size' column.
    """
    df['company_size'] = df.apply(apply_company_size, axis=1)
    df = df.drop(columns=['company_size_large_size_business', 'company_size_mid_size_business', 'company_size_small_business', 'company_size_not_available'])
    return df


def apply_market_type(row):
    """
    Assign the market type category based on the presence of a specific column.

    Parameters:
        row (pd.Series): A row of the DataFrame.

    Returns:
        int: The numerical value representing the market type category.
    """
    if row['market_na'] == 1:
        return MarketType.NA.value
    elif row['market_latam'] == 1:
        return MarketType.LATAM.value
    elif row['market_emea'] == 1:
        return MarketType.EMEA.value
    elif row['market_aus_row'] == 1:
        return MarketType.AUS_ROW.value
    else:
        return MarketType.NOT_AVAILABLE.value  # Default to 0 for 'Unknown'


def transform_market_type(df):
    """
    Transform the market type columns into a single 'market_type' column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'market_type' column.
    """
    df['market_type'] = df.apply(apply_market_type, axis=1)
    df = df.drop(columns=['market_na', 'market_latam', 'market_emea', 'market_aus_row'])
    return df


def transform_start_end_datetimes(df):
    """
    Transform the contract start and end date columns into new features.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new contract date-related features.
    """
    # Convert the 'contract_start_date' and 'contract_end_date' columns to timestamps
    df['contract_start_date'] = pd.to_datetime(df['contract_start_date'])
    df['contract_end_date'] = pd.to_datetime(df['contract_end_date'])

    # Extract year, month, day, and dayofyear components for both columns
    df['contract_start_year'] = df['contract_start_date'].dt.year
    df['contract_start_month'] = df['contract_start_date'].dt.month
    df['contract_start_day'] = df['contract_start_date'].dt.day
    df['contract_start_dayofyear'] = df['contract_start_date'].dt.dayofyear

    df['contract_end_year'] = df['contract_end_date'].dt.year
    df['contract_end_month'] = df['contract_end_date'].dt.month
    df['contract_end_day'] = df['contract_end_date'].dt.day
    df['contract_end_dayofyear'] = df['contract_end_date'].dt.dayofyear

    # Calculate the contract length in seconds
    df['contract_length_in_seconds'] = (df['contract_end_date'] - df['contract_start_date']).dt.total_seconds()

    # CLV
    df['customer_lifetime_value'] = df['contract_length_in_seconds'] * df['amount_due_usd'] * df['quantity_due']

    df = df.drop(columns=['contract_end_date', 'contract_start_date'])
    return df


def transforms(df):
    """
    Perform a series of data transformations on the input DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df = df.sort_values('contract_start_date')
    df = transform_company_size(df)
    df = transform_market_type(df)
    df = transform_start_end_datetimes(df)
    df = df.drop(columns=['Account_id'])
    return df


def train_test_split_df(df, test_size=0.2, random_state=None, even=False):
    """
    Perform a train-test split on the input DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        test_size (float): The proportion of the data to include in the test split.
        random_state (int, optional): Seed for the random number generator.
        even (bool, optional): If True, perform stratified split for an even distribution of categories in the target variable.

    Returns:
        pd.DataFrame: X_train (features for training).
        pd.DataFrame: X_test (features for testing).
        pd.Series: y_train (target variable for training).
        pd.Series: y_test (target variable for testing).
    """
    X, y = df.drop(columns=['churn']), df['churn']
    stratify = y if even else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    return X_train, X_test, y_train, y_test


def fit_shap(X, y, save_plot=False, save_path=None):
    """
    Train a Machine Learning Model (XGBoost) and compute SHAP values for feature importance.

    Parameters:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        visualize (bool, optional): If True, create a feature importance plot (Summary Plot).
        save_plot (bool, optional): If True, save the plot instead of displaying it.
        save_path (str, optional): The path to save the plot. Required if save_plot is True.

    Returns:
        xgb.XGBClassifier: The trained XGBoost model.
        shap.Explainer: The SHAP explainer object.
    """
    model = xgb.XGBClassifier()
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)

    average_shap_values = pd.DataFrame(shap_values).mean()
    shap.summary_plot(shap_values, X, plot_type="bar", feature_names=X.columns)
    if save_plot and save_path:
        plt.savefig(save_path,format='png')  # Save the plot to the specified path
    return model, explainer


def run(disallowed_features=None, save_plot=False, save_path=''):
    """
    Load data, transform it, and train models with and without stratified train-test split.

    """
    df = pd.read_csv('https://github.com/CharlieSergeant/goto-project/raw/main/data/raw/Sample%20Data%20for%20Interview%20Project.csv')
    transformed_df = transforms(df)

    if disallowed_features is not None:
        for disallowed_column in disallowed_features:
            if disallowed_column in transformed_df.columns:
                transformed_df = transformed_df.drop(columns=[disallowed_column])

    print('Running Unstratified fit_shap test...')
    unstratified_X_train, unstratified_X_test, unstratified_y_train, unstratified_y_test = train_test_split_df(transformed_df, test_size=0.2, random_state=42, even=False)
    unstratified_train_model, unstratified_shap = fit_shap(unstratified_X_train, unstratified_y_train, save_plot=save_plot, save_path=f'{save_path}unstratified_shap_chart.png')
    unstratified_accuracy = unstratified_train_model.score(unstratified_X_test, unstratified_y_test)
    print("Unstratified Model Accuracy:", unstratified_accuracy)

    print('Running Stratified fit_shap test...')
    stratified_X_train, stratified_X_test, stratified_y_train, stratified_y_test = train_test_split_df(transformed_df, test_size=0.2, random_state=42, even=True)
    stratified_train_model, stratified_shap = fit_shap(stratified_X_train, stratified_y_train, save_plot=save_plot, save_path=f'{save_path}stratified_shap_chart.png')
    stratified_accuracy = stratified_train_model.score(stratified_X_test, stratified_y_test)
    print("Stratified Model Accuracy:", stratified_accuracy)


if __name__ == "__main__":
    run(
        disallowed_features=None,
        save_plot=True,
        save_path='./data/output/'
    )