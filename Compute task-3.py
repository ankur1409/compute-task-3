import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data2.csv')

# Convert categorical variables to numerical variables(isse you can easily train the data)
df = pd.get_dummies(df, columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country'])

# Splitting  data into training and testing sets(this is done to train data as supervised learning)
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trains logistic regression model for the data
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# this makes predictions
y_pred = logreg.predict(X_test)

# this evaluates model ka performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
