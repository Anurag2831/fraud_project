import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest  # Anomaly Detection

# Load your fraud data (replace 'fraud_data.csv' with your filename)
data = pd.read_csv("fraud_data.csv")

# Separate features and target variable
features = data.drop('label', axis=1)  # Assuming 'label' is your fraud indicator
target = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Anomaly Detection Model (Isolation Forest)
model = IsolationForest(contamination=0.01)  # Adjust contamination for anomaly %

# Train the model
model.fit(X_train)

# Predict on testing data (1: normal, -1: anomaly)
predictions = model.predict(X_test)

# Evaluate model performance (metrics like precision, recall)
# ...

# Real-time scenario (replace with your logic)
new_transaction = # Your new transaction data

prediction = model.predict(new_transaction.reshape(1, -1))

if prediction == -1:
  print("Potential Fraudulent Transaction!")
else:
  print("Transaction seems legitimate.")
