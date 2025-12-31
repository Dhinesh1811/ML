🚗 Uber Rides Prediction using Machine Learning
        
        This project predicts weekly Uber ride demand using a Machine Learning model trained on historical taxi data and deployed through a Flask web application.

📌Project Description
        
        The ML model is trained using the dataset taxi.csv
        The trained model is used to predict weekly Uber rides
        A Flask-based web interface allows users to enter inputs and get predictions

🧠 Inputs Used for Prediction
        
        Price per week – Average weekly Uber fare
        Population – City population
        Monthly income – Average monthly income
        Average parking per month – Parking cost
        
Steps to start the application:

        Execute all the cells in ML_Model.ipynb to create the Model
        python -m venv myenv
        myenv\scripts\activate.bat
        pip install poetry
        poetry config virtualenvs.create false
        poetry init
                <img width="1077" height="576" alt="image" src="https://github.com/user-attachments/assets/cf44be75-8883-494e-a939-104b028fdb11" />
        poetry add numpy==2.2.6 pandas==2.3.3 flask==3.1.2 scikit-learn==1.4.2 ipykernel
        python app.py
        Lauch http://127.0.0.1:5000 to predict the weekly rides
