import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive')
df = pd.read_csv('gdrive/My Drive/Colab Notebooks/Machine Learning Notes/DataSets/nba_logreg.csv')

df.head()

df.info()

df.isnull().sum()

plt.figure(figsize=(6,4))
sns.countplot(x='TARGET_5Yrs', data=df)
plt.title('Players with Career ≥5 Years vs <5 Years')
plt.xlabel('5+ Year Career (1=Yes, 0=No)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['PTS'], bins=30, kde=True)
plt.title('Distribution of Points per Game')
plt.xlabel('Points per Game')
plt.ylabel('Frequency')
plt.show()

features_to_plot = ['PTS', 'AST', 'REB', 'TOV', 'TARGET_5Yrs']
sns.pairplot(df[features_to_plot], hue='TARGET_5Yrs', diag_kind='kde', corner=True)
plt.suptitle('Pairwise Relationships of Selected Features', y=1.02)
plt.show()

df_clean = df.dropna(subset=['3P%'])

feature_cols = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3P%', 'FTM', 'FTA', 'FT%',
                'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']
X = df_clean[feature_cols]
y = df_clean['TARGET_5Yrs']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': logreg.coef_[0]
})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Feature Importance from Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

