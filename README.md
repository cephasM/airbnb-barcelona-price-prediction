Airbnb Barcelona Price Prediction

This project is a machine learning pipeline for predicting Airbnb listing prices in Barcelona. It involves data preprocessing, model training, hyperparameter optimization, and performance evaluation using a Random Forest Regressor.

📁 Dataset

The dataset used is a CSV file containing Airbnb listings in Barcelona.

Path: data/airbnb_barcelona.csv

🔧 Project Structure

Data Cleaning: Unnecessary columns are dropped, and price values are cleaned and filtered.

Feature Engineering: Categorical and numerical features are selected and preprocessed.

Modeling: A RandomForestRegressor is trained inside a Pipeline.

Hyperparameter Tuning: Uses GridSearchCV for model optimization.

Evaluation: RMSE and R^2 metrics are reported; a scatter plot compares actual vs predicted prices.

📊 Features Used

host_is_superhost

property_type

room_type

latitude, longitude

accommodates, bathrooms, bedrooms, beds

minimum_nights

review_scores_rating

neighbourhood

availability_365, number_of_reviews_ltm

🧪 Model Evaluation

Initial RMSE: ~51.31

Initial R² Score: ~0.63

Optimized RMSE: ~51.34

Optimized R² Score: ~0.63

📈 Visualization

The model's performance is visualized with a scatter plot of actual vs predicted prices.

💡 How to Run

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn

Run the script:

python airbnb_price_prediction.py

📎 File

airbnb_price_prediction.py: Main Python script.

📌 Notes

Price values were limited to below $600 to remove outliers.

Missing values are handled using SimpleImputer.

OneHotEncoding is used for categorical variables.

📬 Contact

For any questions or suggestions contact https://www.linkedin.com/in/pierre-musili/
