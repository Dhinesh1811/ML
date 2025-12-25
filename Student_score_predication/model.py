from sklearn.preprocessing import MinMaxScaler
import Student_score_prediction

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
