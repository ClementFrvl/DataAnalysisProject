import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from model_recaps import models_info

df_base = pd.read_csv('https://raw.githubusercontent.com/ClementFrvl/Datasets/main/data.csv', sep=';')
df = pd.read_csv('https://raw.githubusercontent.com/ClementFrvl/Datasets/main/data_filtered.csv')
df_model = pd.read_csv('https://raw.githubusercontent.com/ClementFrvl/Datasets/main/data_modeling.csv')

# Numerical columns
base_num_cols = df_base.select_dtypes(include=['number']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
discrete_cols = ['Seasons', 'Holiday', 'Year', 'Month', 'Day', 'Hour']

models = [LinearRegression(), XGBRegressor(), Ridge(), Lasso(), KNeighborsRegressor(), DecisionTreeRegressor(),
          RandomForestRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(), SVR(), LGBMRegressor()]
scaling_methods = [StandardScaler(), MinMaxScaler(), RobustScaler()]

st.set_page_config(
    page_title='Seoul Bike Rental Data Analysis',
    layout='wide'
)

st.title("Seoul Bike Rental Data Analysis")

placeholder = st.empty()

fig_size = (10, 5)

with (placeholder.container()):

    st.subheader("Data Overview on base dataset")

    col_0, col_01 = st.columns(2)
    col_0.subheader("Variable distribution")
    col_01.subheader("Variable correlation to target")
    select_0 = col_0.selectbox("Base variable distribution", df_base.columns.tolist(), index=0)
    select_01 = col_01.selectbox("Base variable correlation", base_num_cols, index=5)

    fig_col0, fig_col01 = st.columns(2)

    with fig_col0:
        fig, ax_col0 = plt.subplots(figsize=fig_size)

        if df_base[select_0].dtype == 'O':
            sns.countplot(x=select_0, data=df_base, ax=ax_col0)
            ax_col0.set_title(f'Count of {select_0}')
        else:
            sns.histplot(df_base, x=select_0, kde=True, ax=ax_col0)
            ax_col0.set_title(f'Distribution of {select_0}')
        ax_col0.legend()
        st.pyplot(fig)

    with fig_col01:
        fig, ax_col01 = plt.subplots(figsize=fig_size)
        sns.regplot(x=select_01, y='Rented Bike Count', data=df_base, ax=ax_col01)
        ax_col01.set_title(f'Regression plot between {select_01} and Rented Bike Count')
        st.pyplot(fig)

    st.subheader("Data Overview on filtered dataset")

    col1, col2 = st.columns(2)
    col1.subheader("Numerical variables")
    col2.subheader("Discrete variables")
    select1 = col1.selectbox("Numerical base distribution", numerical_cols, index=2)
    select2 = col2.selectbox("Discrete variable distribution", discrete_cols, index=3)

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        fig, ax_col1 = plt.subplots(figsize=fig_size)
        sns.histplot(df, x=select1, kde=True, ax=ax_col1)
        ax_col1.axvline(df[select1].mean(), color='r', linestyle='dashed', linewidth=2, label='Mean')
        ax_col1.axvline(df[select1].median(), color='b', linestyle='dashed', linewidth=2, label='Median')
        ax_col1.set_title(f'Distribution of {select1}')
        ax_col1.legend()
        st.pyplot(fig)

    with fig_col2:
        fig, ax_col2 = plt.subplots(figsize=fig_size)
        df.groupby(select2)['Rented Bike Count'].sum().reset_index().plot.bar(x=select2, y='Rented Bike Count', ax=ax_col2)
        ax_col2.set_title(f'Number of bikes rented per {select2}')
        for p in ax_col2.patches:
            ax_col2.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                             textcoords='offset points')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    select3 = col3.selectbox("Select discrete variable (x)", discrete_cols, index=5)
    select4 = col4.selectbox("Select discrete variable (hue)", discrete_cols, index=0)

    fig_col3 = st.columns(1)
    fig, ax_col3 = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=df, x=select3, y='Rented Bike Count', hue=select4, ax=ax_col3, marker='x', markeredgecolor="black")
    ax_col3.set_title(f'Lineplot of {select3} and Rented Bike Count, hued by {select4}')
    ax_col3.legend(loc='upper left')
    st.pyplot(fig)

    st.subheader("Feature engineering")

    col5, = st.columns(1)
    select5 = col5.selectbox("Select numerical variable", numerical_cols, index=0)

    fig_col4, fig_col5= st.columns(2)

    with fig_col4:
        fig, ax_col1 = plt.subplots(figsize=fig_size)
        sns.histplot(df[select5], kde=True, ax=ax_col1)
        ax_col1.axvline(df[select5].mean(), color='r', linestyle='dashed', linewidth=2, label='Mean')
        ax_col1.axvline(df[select5].median(), color='b', linestyle='dashed', linewidth=2, label='Median')
        ax_col1.set_title(f'Base data distribution of {select1}')
        ax_col1.legend()
        st.pyplot(fig)

    with fig_col5:
        fig, ax_col1 = plt.subplots(figsize=fig_size)
        sns.histplot(np.log1p(df[select5]), kde=True, ax=ax_col1)
        ax_col1.axvline(np.log1p(df[select5]).mean(), color='r', linestyle='dashed', linewidth=2, label='Mean')
        ax_col1.axvline(np.log1p(df[select5]).median(), color='b', linestyle='dashed', linewidth=2, label='Median')
        ax_col1.set_title(f'Log transformation of {select1}')
        ax_col1.legend()
        st.pyplot(fig)

    fig_col6, fig_col7 = st.columns(2)

    with fig_col6:
        fig, ax_col1 = plt.subplots(figsize=fig_size)
        sns.histplot(np.sqrt(df[select5]), kde=True, ax=ax_col1)
        ax_col1.axvline(np.sqrt(df[select5]).mean(), color='r', linestyle='dashed', linewidth=2, label='Mean')
        ax_col1.axvline(np.sqrt(df[select5]).median(), color='b', linestyle='dashed', linewidth=2, label='Median')
        ax_col1.set_title(f'Square root transformation of {select1}')
        ax_col1.legend()
        st.pyplot(fig)

    with fig_col7:
        fig, ax_col1 = plt.subplots(figsize=fig_size)
        sns.histplot(np.cbrt(df[select5]), kde=True, ax=ax_col1)
        ax_col1.axvline(np.cbrt(df[select5]).mean(), color='r', linestyle='dashed', linewidth=2, label='Mean')
        ax_col1.axvline(np.cbrt(df[select5]).median(), color='b', linestyle='dashed', linewidth=2, label='Median')
        ax_col1.set_title(f'Cube root transformation of {select1}')
        ax_col1.legend()
        st.pyplot(fig)

    st.subheader("Modeling")

    col_model, col_scaler = st.columns(2)
    select_model = col_model.selectbox("Select model", [m.__class__.__name__ for m in models], index=0)
    select_scaler = col_scaler.selectbox("Select scaler", [m.__class__.__name__ for m in scaling_methods], index=0)

    fig_col8, res_col = st.columns(2)

    with fig_col8:
        X = df_model.drop('Rented Bike Count', axis=1)
        y = df_model['Rented Bike Count']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if select_scaler is not None:
            scaler = scaling_methods[[s.__class__.__name__ for s in scaling_methods].index(select_scaler)]
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
        model = models[[m.__class__.__name__ for m in models].index(select_model)]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        fig_8, ax = plt.subplots(figsize=fig_size)
        plt.scatter(y_pred, y_test, alpha=0.2)
        plt.xlim(0, 40)
        plt.ylim(0, 40)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Predicted vs Actual values of {select_model}')
        st.pyplot(fig_8)

    with res_col:
        st.write(f'R2 score: {r2:.3f}')
        st.write(f'MSE: {mse:.3f}')
        st.write(f'MAE: {mae:.3f}')

        st.write("Model description: " + models_info[select_model]['description'])
        st.write("Model key feature: " + models_info[select_model]['key_feature'])



