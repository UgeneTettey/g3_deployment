import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# model = pickle.load(open("C:/Users/Kelvin/OneDrive/Documents/iiAfrica/Medical Price Prediction/random_forest_model.pkl",'rb'))
# model = pickle.load(open("C:/Users/Eugene/Desktop/iiAfrica/IIP_2024/ML/projects/G3/from_group/random_forest_model.pkl", 'rb'))
model = pickle.load(open("random_forest_model.pkl", 'rb'))

# categorical features
categorical_features = {
    'sex':['male','female'],
    'smoker':['yes', 'no'],
    'region':['southwest', 'southeast', 'northwest', 'northeast']
}

# define ecndoder dictionary
encoder_dict = {feature: LabelEncoder().fit(values) for feature, values in categorical_features.items()}


def main():
    st.set_page_config(page_title="Predicting Medical Costs", page_icon=":bar_chart:", layout="centered", initial_sidebar_state="expanded")
    st.title("Medical Price Prediction")
    st.sidebar.header('Patient/Client Details')

    input_data = {}

    #input variables
    input_data['age'] = st.sidebar.number_input("Age", step=1)
    input_data['bmi'] = st.sidebar.number_input("BMI")
    input_data['sex'] = st.sidebar.selectbox("Gender", options=categorical_features['sex'])
    input_data['children'] = st.sidebar.number_input("Number of children", step=1)
    input_data['smoker'] = st.sidebar.selectbox("Are you a smoker?", options=categorical_features['smoker'])
    input_data['region'] = st.sidebar.selectbox("Which region do you come from?", options=categorical_features['region'])

    input_df = pd.DataFrame([input_data])

    # Encode categorical feature
    for feature, encoder in encoder_dict.items():
        input_df[feature] = encoder.transform(input_df[feature])

    #prediction
    if st.button("Predict"):
        makeprediction = model.predict(input_df)
        output = round(makeprediction[0],2)
        st.success('Your medical cost {}'.format(output))

if __name__=='__main__':
    main()