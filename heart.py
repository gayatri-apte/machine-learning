import pandas as pd
import numpy as np 
from sklearn import preprocessing 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load and clean dataset
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(columns=['education'], inplace=True)
disease_df.rename(columns={'male': 'sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)  # Remove rows with missing values

# Check class distribution
print(disease_df)
print(disease_df.TenYearCHD.value_counts())

# Select features and scale them
X = np.asarray(disease_df[['age', 'sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Plot class balance
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df, palette="BuGn_r")
plt.show()

# Basic trend line plot for the target column
disease_df['TenYearCHD'].plot()
plt.show()

# Logistic Regression model training and prediction
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Evaluate accuracy
from sklearn.metrics import accuracy_score
print("Accuracy of model is =", accuracy_score(y_test, y_pred))

# Print detailed classification metrics
from sklearn.metrics import confusion_matrix, classification_report
print("The details of confusion matrix are=")
print(classification_report(y_test, y_pred))

# Plot confusion matrix as heatmap
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, columns=['predicted:0', 'predicted:1'], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.show()
