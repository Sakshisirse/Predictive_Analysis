import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_excel("customer_support_data_.csv")
print(df.head())
print(df.info())
df_cleaned = df.dropna()
df_cleaned["IssueType"] = df_cleaned["IssueType"].astype("category").cat.codes
features = ["ResponseTime", "IssueType", "AgentExperience"]
target = "ResolutionTime"
X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned[target], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMAE: {mae}\nR² Score: {r2}")
new_data = pd.DataFrame({"ResponseTime": [5], "IssueType": [2], "AgentExperience": [3]})
predicted_resolution_time = model.predict(new_data)

print(f"Predicted Resolution Time: {predicted_resolution_time[0]} hours")
df_cleaned.to_csv("processed_customer_support_data.csv", index=False)