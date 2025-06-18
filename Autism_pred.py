import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

# Load dataset
df = pd.read_csv("train.csv")
print(df.head())
print(df.shape)
df.info()
print(df.describe().T)

# Check column names
print(df.columns)

# Fix categorical replacements
df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})

# Show ASD distribution
plt.pie(df['Class/ASD'].value_counts().values, labels=df['Class/ASD'].value_counts().index, autopct='%1.1f%%')
plt.title("ASD Class Distribution")
plt.show()

# Separate columns by type
ints = []
objects = []
floats = []

for col in df.columns:
    if df[col].dtype == int:
        ints.append(col)
    elif df[col].dtype == object:
        objects.append(col)
    else:
        floats.append(col)

# Drop non-useful columns
for col in ['ID', 'Class/ASD']:
    if col in ints:
        ints.remove(col)

# Melt and plot integer feature distributions
df_melted = df.melt(id_vars=['ID', 'Class/ASD'], value_vars=ints, var_name='col', value_name='value')
plt.subplots(figsize=(15, 15))
for i, col in enumerate(ints):
    plt.subplot(5, 3, i + 1)
    sb.countplot(x='value', hue='Class/ASD', data=df_melted[df_melted['col'] == col])
plt.tight_layout()
plt.show()

# Categorical object feature distributions
plt.subplots(figsize=(15, 15))
for i, col in enumerate(objects):
    plt.subplot(5, 3, i + 1)
    sb.countplot(x=col, hue='Class/ASD', data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Fix for column name
if 'country_of_res' in df.columns:
    plt.figure(figsize=(15, 5))
    sb.countplot(data=df, x='country_of_res', hue='Class/ASD')
    plt.xticks(rotation=90)
    plt.show()

# Plot float distributions
plt.subplots(figsize=(15, 5))
for i, col in enumerate(floats):
    plt.subplot(1, 2, i + 1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.show()

# Boxplots for floats
plt.subplots(figsize=(15, 5))
for i, col in enumerate(floats):
    plt.subplot(1, 2, i + 1)
    sb.boxplot(y=df[col])
plt.tight_layout()
plt.show()

# Filter out invalid results
df = df[df['result'] > -5]

# Convert age to group
def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

df['ageGroup'] = df['age'].apply(convertAge)
sb.countplot(x=df['ageGroup'], hue=df['Class/ASD'])
plt.show()

# Add feature
def add_feature(data):
    data['sum_score'] = 0
    for col in data.loc[:, 'A1_Score':'A10_Score'].columns:
        data['sum_score'] += data[col]
    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
    return data

df = add_feature(df)
sb.countplot(x=df['sum_score'], hue=df['Class/ASD'])
plt.show()

# Log transform age
df['age'] = df['age'].apply(lambda x: np.log(x))
sb.histplot(df['age'], kde=True)
plt.show()

# Encode object labels
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data

df = encode_labels(df)

# Correlation heatmap
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.title("Correlation > 0.8")
plt.show()

# Model training
removal = ['ID', 'age_desc', 'used_app_before', 'austim']
features = df.drop(removal + ['Class/ASD'], axis=1)
target = df['Class/ASD']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

models = [LogisticRegression(max_iter=1000), XGBClassifier(use_label_encoder=False, eval_metric='logloss'), SVC(kernel='rbf')]

for model in models:
    model.fit(X, Y)
    print(f'{model.__class__.__name__} :')
    print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))
    print()
