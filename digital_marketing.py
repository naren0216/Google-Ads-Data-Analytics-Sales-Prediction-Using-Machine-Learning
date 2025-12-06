# ------------------------------------------
# IMPORT ALL REQUIRED LIBRARIES
# ------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------
# LOAD AND CLEAN DATA
# ------------------------------------------
df = pd.read_csv("C:/Users/Naren/Desktop/GoogleAds_DataAnalytics_Sales_Uncleaned.csv")

df.columns = df.columns.str.strip()

df = df.dropna(subset=['Clicks', 'Impressions', 'Cost', 'Conversions', 'Sale_Amount'])

df['Cost'] = df['Cost'].replace('[\₹\$,]', '', regex=True).astype(float)
df['Sale_Amount'] = df['Sale_Amount'].replace('[\₹\$,]', '', regex=True).astype(float)

# ------------------------------------------
# CREATE NEW METRICS
# ------------------------------------------
df['CTR (%)'] = (df['Clicks'] / df['Impressions']) * 100
df['CPC'] = df['Cost'] / df['Clicks']
df['Conversion Rate (%)'] = (df['Conversions'] / df['Clicks']) * 100
df['ROAS'] = df['Sale_Amount'] / df['Cost']

# ------------------------------------------
# SUMMARY STATISTICS
# ------------------------------------------
print("\n--- DATA SUMMARY ---")
print(df[['Clicks', 'Impressions', 'Cost', 'Conversions', 'Sale_Amount', 
          'CTR (%)', 'CPC', 'Conversion Rate (%)', 'ROAS']].describe())

# ------------------------------------------
# VISUALIZATIONS
# ------------------------------------------

# CTR by campaign
plt.figure(figsize=(10, 5))
sns.barplot(x='Campaign_Name', y='CTR (%)', data=df)
plt.xticks(rotation=45)
plt.title('Click Through Rate by Campaign')
plt.tight_layout()
plt.show()

# CPC distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['CPC'], bins=15, kde=True, color='orange')
plt.title('Cost Per Click Distribution')
plt.tight_layout()
plt.show()

# Revenue vs Cost
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Cost', y='Sale_Amount', data=df, hue='Campaign_Name', s=100)
plt.title('Sale Amount vs Cost')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df[['Clicks', 'Impressions', 'Cost', 'Conversions', 'Sale_Amount',
                'CTR (%)', 'CPC', 'ROAS']].corr(), annot=True, cmap='coolwarm')
plt.title('Metric Correlations')
plt.tight_layout()
plt.show()

# ------------------------------------------
# MACHINE LEARNING MODEL (PREDICT REVENUE)
# ------------------------------------------
features = ['Clicks', 'Impressions', 'Cost', 'Conversions', 'CTR (%)', 'CPC', 'Conversion Rate (%)']
X = df[features]
y = df['Sale_Amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ------------------------------------------
# MODEL EFFICIENCY METRICS
# ------------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- MODEL EFFICIENCY ---")
print(f"R² Score: {round(r2, 4)}")
print(f"MAE: {round(mae, 4)}")
print(f"RMSE: {round(rmse, 4)}")

# ------------------------------------------
# PREDICTION VISUALIZATIONS
# ------------------------------------------

# A. Actual vs Predicted Revenue
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', color='red')
plt.xlabel("Actual Revenue (Sale Amount)")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.tight_layout()
plt.show()

# B. Residual Plot
errors = y_test - y_pred
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_pred, y=errors)
plt.axhline(0, color='red')
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

# C. Error Distribution
plt.figure(figsize=(7, 6))
sns.histplot(errors, bins=20, kde=True)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.tight_layout()
plt.show()

# D. Sorted Actual vs Predicted Line Plot
sorted_idx = np.argsort(y_test.values)
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[sorted_idx], label='Actual', linewidth=3)
plt.plot(y_pred[sorted_idx], label='Predicted', linestyle='--')
plt.xlabel("Samples")
plt.ylabel("Revenue (Sale Amount)")
plt.title("Actual vs Predicted Revenue (Sorted)")
plt.legend()
plt.tight_layout()
plt.show()
