# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/airbnb_barcelona.csv")
df.head()

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'id', 'host_id'], inplace=True, errors='ignore')

# Clean and convert the 'price' column
df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)

# Filter out extreme price values
df = df[(df["price"] > 0) & (df["price"] < 600)]

# Define features and target
features = [
    'host_is_superhost', 'property_type', 'room_type', 'latitude', 'longitude',
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights', 
    'review_scores_rating', 'neighbourhood', 'availability_365', 'number_of_reviews_ltm'
]
target = 'price'

X = df[features]
y = df[target]

# Separate feature types
categorical = ['host_is_superhost', 'property_type', 'room_type', 'neighbourhood']
numerical = [
    'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 
    'beds', 'minimum_nights', 'review_scores_rating', 
    'availability_365', 'number_of_reviews_ltm'
]

# Define preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical)
    ]
)

# Define model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Initial RMSE:", rmse)
print("Initial R² score:", r2)

# Define hyperparameter grid for tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}

# Grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Final evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Best model found by GridSearchCV")
print("Optimized RMSE:", rmse)
print("Optimized R² score:", r2)

# Visual comparison between actual and predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Optimized Model)")
plt.legend()
plt.grid(True)
plt.show()
