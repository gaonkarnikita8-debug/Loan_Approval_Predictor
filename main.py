import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

#1.load the data
df = pd.read_csv("Loan_Approval_Predictor/training_data.csv")

#2. Understand the data
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

#3.find missing values
missing_values = df.isnull().sum()
# print(missing_values)
#Loan_ID               0
# Gender               13
# Married               3
# Dependents           15
# Education             0
# Self_Employed        32
# ApplicantIncome       0
# CoapplicantIncome     0
# LoanAmount           22
# Loan_Amount_Term     14
# Credit_History       50
# Property_Area         0

#4.clean the data
#4.1. to remove all rows with missing data
# new_df = df.dropna()
# print(new_df)



#4.2. OR to give some default values to missing data

#for colums 'Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term' use mode (most common values)
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term']:
    df[col] = df[col].fillna(df[col].mode()[0])

#dependents are str format , need to update as int

df["Dependents"] = df["Dependents"].replace("3+", "3")
df["Dependents"] = df["Dependents"].astype(int)

#as credit history is 0/1 for missing values fill it as -1
df['Credit_History'] = df['Credit_History'].fillna(-1)

#for loan amount fill median as mean can create big difference
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
#check again if any values missing from new_df
# print(new_df.isnull().sum())

#check and remove duplicates from LOAN_ID column
new_df = df.drop_duplicates(subset="Loan_ID")

#convert text to numbers
new_df['Education'] = new_df['Education'].map({'Not Graduate' : 0, 'Graduate' : 1})
new_df['Gender'] = new_df['Gender'].map({'Male' : 0, 'Female' : 1})
new_df['Married'] = new_df['Married'].map({'No' : 0, 'Yes' : 1})
new_df['Self_Employed'] = new_df['Self_Employed'].map({'No' : 0, 'Yes' : 1})
new_df['Property_Area'] = new_df['Property_Area'].map({'Rural' : 0, 'Semiurban' : 1, 'Urban' : 2})
new_df['Loan_Status'] = new_df['Loan_Status'].map({'N' : 0, 'Y' : 1})

#save the cleaned data to new csv
# new_df.to_csv("Loan_Approval_Predictor/cleaned_train_data.csv")

#EDA
new_df['Credit_History'].value_counts().plot(kind='bar')
plt.title('Graph')
plt.xlabel('Credit_History')
plt.ylabel('No of Applicants')
plt.show()

print(new_df['Loan_Status'].value_counts())
# print(pd.crosstab(new_df['Credit_History'], new_df['Loan_Status']))
print(pd.crosstab(new_df['Credit_History'], new_df['Loan_Status'],normalize='index') * 100)

print(pd.crosstab(new_df['Married'], new_df['Loan_Status'],normalize='index') * 100)

print(pd.crosstab(new_df['Education'], new_df['Loan_Status'],normalize='index') * 100)

print(pd.crosstab(new_df['Property_Area'], new_df['Loan_Status'],normalize='index') * 100)
print(pd.crosstab(new_df['LoanAmount'], new_df['Loan_Status'],normalize='index') * 100)


# df["Income_Group"] = pd.cut(
#     df["ApplicantIncome"],
#     bins=[0, 3000, 5000, 10000, float("inf")],
#     labels=["Low", "Medium", "High", "Very High"]
# )

# print(pd.crosstab(
#     df["Income_Group"],
#     df["Loan_Status"],
#     normalize="index"
# ) * 100)



plt.figure(figsize=(8, 5))

plt.hist(df["LoanAmount"].dropna(), bins=30)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Number of Applicants")

plt.show()

#5. Select X and y [drop target= Loan Status]
X = new_df.drop(['Loan_Status','Loan_ID'], axis=1)
y = new_df['Loan_Status']

# print('X shape: ', X.shape)
# print('y shape: ', y.shape)

# print(X.head())
# print(y.head())




#5.split data
#random_state means whenevr it is run it chooses same random nos, test size = 20%, stratify = y means both training and testing data has same proportion
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=0.2,stratify=y)

# print('X shape: ', X_train.shape)
# print('y shape: ', y_train.shape)

# print('X shape: ', X_test.shape)
# print('y shape: ', y_test.shape)

# print("Training data:")
# print(y_train.value_counts())

# print("\nTesting data:")
# print(y_test.value_counts())

#scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#1.train your model- Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# predict based on different ML models
y_predict = model.predict(X_test_scaled)
print(y_predict)

# compare values
compare_values = pd.DataFrame({"Actual Values": y_test.values, "Predicted Values" : y_predict})

print(compare_values.head(20))

#calculate accuracy
accurate_score_lr = accuracy_score(y_test, y_predict)
print("Accuracy percentage : ",accurate_score_lr* 100)

cm = confusion_matrix(y_test, y_predict)
print(cm)

print(classification_report(y_test, y_predict))

#2.train model - Decision tree
tree_model = DecisionTreeClassifier(random_state=50)

tree_model.fit(X_train, y_train)
y_predict_tree = tree_model.predict(X_test)
print(y_predict_tree)

compare_values = pd.DataFrame({"Actual Values": y_test.values, "Predicted Values" : y_predict_tree})
print(compare_values)

#calculate accuracy
accurate_score_dt = accuracy_score(y_test, y_predict_tree)
print("Accuracy percentage : ",accurate_score_dt * 100)

cm_tree = confusion_matrix(y_test, y_predict_tree)
print(cm_tree)

print(classification_report(y_test, y_predict_tree))

ConfusionMatrixDisplay.from_predictions(y_test, y_predict_tree, display_labels= ['Rejected', 'Approved'], cmap = 'Blues')
plt.title("COnfusion Matrix : Decision Tree")
plt.show()

#3.model training - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=50)

rf_model.fit(X_train, y_train)

#predict
y_predict_rf = rf_model.predict(X_test)

accuracy_score_rf = accuracy_score(y_test, y_predict_rf)
print("Accuracy percentage : ",accuracy_score_rf * 100)

print("Model Performance")
print("-------------------------")
print(f"Logistic Regression: {accurate_score_lr:.4f}")
print(f"Decision Tree:       {accurate_score_dt:.4f}")
print(f"Random Forest:       {accuracy_score_rf:.4f}")

feature_importance = pd.DataFrame({'Features' : X_train.columns, 'Importance' : rf_model.feature_importances_})

feature_importance = feature_importance.sort_values(by='Importance', ascending= False)
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Features"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Features Importance")

plt.gca().invert_yaxis()

plt.show()


#grid tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best Parameters:")
print(rf_grid.best_params_)

print("Best Cross-Validation Score:")
print(rf_grid.best_score_)

best_rf_model = rf_grid.best_estimator_

y_pred_tuned = best_rf_model.predict(X_test)

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print("Tuned Random Forest Test Accuracy:", tuned_accuracy)
print("Tuned Random Forest Test Accuracy (%):", tuned_accuracy * 100)

print("Tuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_tuned))


ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_tuned,
    display_labels=["Rejected", "Approved"],
    cmap="Blues"
)

plt.title("Tuned Random Forest - Confusion Matrix")
# plt.show()

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "Accuracy": [
        accurate_score_lr,
        accurate_score_dt,
        accuracy_score_rf,
        tuned_accuracy
    ]
})

model_results = model_results.sort_values(
    by="Accuracy",
    ascending=False
)

print(model_results)

plt.figure(figsize=(10, 6))

plt.bar(
    model_results["Model"],
    model_results["Accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=20)
plt.ylim(0, 1)

plt.show()

joblib.dump(best_rf_model, 'Loan_Approval_Prediction_model.pkl')