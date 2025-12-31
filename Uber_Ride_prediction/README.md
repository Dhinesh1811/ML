
Steps:
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