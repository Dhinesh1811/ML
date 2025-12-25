import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_csv(path):
    """
    Load CSV file and return DataFrame
    """
    return pd.read_csv(path)

def clean_missing_values(df):
    """
    Fill missing numeric values with mean
    """
    return df.fillna(df.mean(numeric_only=True))

def normalize_column(df, col):
    """
    Scale values between 0 and 1
    """
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def split_data(X, y, test_size=0.2):
    """
    Split data into train and test sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


df = load_csv("students_mark.csv")
df = clean_missing_values(df)

df = normalize_column(df, "math_score")
df = normalize_column(df, "science_score")
df = normalize_column(df, "english_score")

X = df[["math_score", "science_score", "english_score"]]
print(X)
y = df["final_score"]
print(y)

X_train, X_test, y_train, y_test = split_data(X, y)

"""
print(X_train.shape, X_test.shape)
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)
"""

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted:", y_pred)
print("Actual:", y_test.values)
print("MSE:", mse)
print("R2 Score:", r2)

print("******")
print(y_pred)
print("******")
print(y_test)
print("******")

df["pass_fail"] = (df["final_score"] >= 80).astype(int)

X = df[["math_score", "science_score", "english_score"]]
y = df["pass_fail"]

X_train, X_test, y_train, y_test = split_data(X, y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

