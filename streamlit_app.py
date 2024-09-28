import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Define the main page for the app
def main():
    # Page title
    st.title("Student Performance Prediction")

    # Sidebar form for input data
    st.sidebar.header('Input Features')

    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    ethnicity = st.sidebar.selectbox('Race/Ethnicity', ['Group A', 'Group B', 'Group C', 'Group D', 'Group E'])
    parental_level_of_education = st.sidebar.selectbox('Parental Level of Education', 
                                                       ['Some high school', 'High school', 'Some college', 
                                                        'Associate’s degree', 'Bachelor’s degree', 'Master’s degree'])
    lunch = st.sidebar.selectbox('Lunch', ['Standard', 'Free/reduced'])
    test_preparation_course = st.sidebar.selectbox('Test Preparation Course', ['None', 'Completed'])
    reading_score = st.sidebar.number_input('Reading Score', min_value=0, max_value=100, value=50)
    writing_score = st.sidebar.number_input('Writing Score', min_value=0, max_value=100, value=50)

    # When the user clicks on "Predict"
    if st.sidebar.button('Predict'):
        # Collecting input data into the CustomData class
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        # Converting input to a DataFrame
        pred_df = data.get_data_as_data_frame()
        st.write("### Input Data")
        st.write(pred_df)

        # Prediction pipeline
        predict_pipeline = PredictPipeline()

        with st.spinner('Predicting...'):
            # Getting predictions
            results = predict_pipeline.predict(pred_df)

        # Displaying the result
        st.success(f"Predicted Math Score: {results[0]}")

if __name__ == "__main__":
    main()
