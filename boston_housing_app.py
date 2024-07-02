import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and data
model = joblib.load('linear_regression_model.joblib')
train_data = pd.read_csv(r"C:\Users\Acer\OneDrive\Desktop\Data Science Project Practice\Housing_Price\train.csv")

# Define the features
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']

st.title('Boston Housing Price Prediction')

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Insights", "Data Exploration"])

if page == "Prediction":
    st.header('Enter House Details:')

    # Create input fields for each feature
    input_data = {}
    for feature in features:
        if feature == 'chas':
            input_data[feature] = st.selectbox(f'{feature.upper()} (Charles River dummy variable)', [0, 1])
        else:
            input_data[feature] = st.number_input(f'Enter {feature.upper()}', value=0.0, format='%f')

    # Data Validation
    if st.button('Validate Input'):
        for feature in features:
            if feature != 'chas':
                min_val, max_val = train_data[feature].min(), train_data[feature].max()
                if input_data[feature] < min_val or input_data[feature] > max_val:
                    st.warning(f'{feature.upper()} value is outside the range of training data. Consider adjusting.')

    # Prediction
    if st.button('Predict House Price'):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f'The predicted house price is: ${prediction * 1000:.2f}')

        # Feature importance for this prediction
        importances = model.coef_ * input_df.values[0]
        feat_importance = pd.DataFrame({'feature': features, 'importance': importances})
        feat_importance = feat_importance.sort_values('importance', key=abs, ascending=False)

        st.subheader('Feature Importance for this Prediction')
        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=feat_importance, ax=ax)
        st.pyplot(fig)

elif page == "Model Insights":
    st.header('Model Insights')

    # Load and display feature importance
    feature_importance = pd.read_csv('feature_importance.csv')
    st.subheader('Overall Feature Importance')
    fig, ax = plt.subplots()
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
    st.pyplot(fig)

    # Display model performance metrics
    st.subheader('Model Performance')
    metrics = joblib.load('model_metrics.joblib')
    st.write(f"R-squared: {metrics['r2']:.3f}")
    st.write(f"Mean Squared Error: {metrics['mse']:.3f}")
    st.write(f"Root Mean Squared Error: {metrics['rmse']:.3f}")

elif page == "Data Exploration":
    st.header('Data Exploration')

    # Display correlation heatmap
    st.subheader('Correlation Heatmap')
    corr = train_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Display distribution of house prices
    st.subheader('Distribution of House Prices')
    fig, ax = plt.subplots()
    sns.histplot(train_data['medv'], kde=True, ax=ax)
    st.pyplot(fig)

    # Allow user to explore relationships between features
    st.subheader('Explore Feature Relationships')
    x_axis = st.selectbox('Choose x-axis feature', features)
    y_axis = st.selectbox('Choose y-axis feature', features)
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_axis, y=y_axis, data=train_data, ax=ax)
    st.pyplot(fig)

# Add descriptions for each feature
st.sidebar.header('Feature Descriptions:')
descriptions = {
    'crim': 'Per capita crime rate by town',
    'zn': 'Proportion of residential land zoned for lots over 25,000 sq.ft.',
    'indus': 'Proportion of non-retail business acres per town',
    'chas': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
    'nox': 'Nitric oxides concentration (parts per 10 million)',
    'rm': 'Average number of rooms per dwelling',
    'age': 'Proportion of owner-occupied units built prior to 1940',
    'dis': 'Weighted distances to five Boston employment centres',
    'rad': 'Index of accessibility to radial highways',
    'tax': 'Full-value property-tax rate per $10,000',
    'ptratio': 'Pupil-teacher ratio by town',
    'black': '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
    'lstat': '% lower status of the population'
}

for feature, description in descriptions.items():
    st.sidebar.write(f'**{feature.upper()}**: {description}')