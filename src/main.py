import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("Educational Practice Project")

# ====== Section: Data Loading ======
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(base_dir, 'housing.csv'))

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn information:")
    print(df.info())
    print("\nStatistics:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())

# ====== Section: Preprocessing. Beginning ======
    median_bedrooms = df['total_bedrooms'].median()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)

    print("\nMissing values after filling:", df.isnull().sum().sum())

    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    print("New features added. Dataset shape:", df.shape)
    print(df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household']].describe())

# ====== Section: Exploratory Data Analysis(EDA) ======
    df.hist(figsize=(15, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "histograms.png"))
    plt.close()

    plt.figure(figsize=(10, 8))

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(images_dir, "correlation_heatmap.png"))
    plt.close()

    plt.figure(figsize=(8, 5))

    sns.histplot(df["median_house_value"], bins=50, kde=True)

    plt.title("House Price Distribution")
    plt.xlabel("Median House Value")
    plt.ylabel("Count")

    plt.savefig(os.path.join(images_dir, "house_price_distribution.png"))
    plt.close()

    print("\n========== EDA ==========")
    print(f"Most correlated with price: median_income")
    print(f"Price range: ${df['median_house_value'].min():,.0f} - ${df['median_house_value'].max():,.0f}")
    print(f"Average price: ${df['median_house_value'].mean():,.0f}")
    
# ====== Section: Preprocessing. Continue ======
    df = pd.get_dummies(df, columns=['ocean_proximity'])

    print("\nColumns after One-Hot Encoding:")
    print(df.columns.tolist())

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nScaling applied.")

# ====== Section: Linear Regression ======
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    lr_predictions = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_predictions)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
    lr_r2 = r2_score(y_test, lr_predictions)

    print("\n========== LINEAR REGRESSION ==========")
    print(f"MAE:  {lr_mae:.2f}")
    print(f"RMSE: {lr_rmse:.2f}")
    print(f"R²:   {lr_r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, lr_predictions, alpha=0.3, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression: Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'lr_plot.png'))
    plt.close()

    print("Plot saved: lr_plot.png")
    
# ====== Section: Random Forest ======
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)

    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_r2 = r2_score(y_test, rf_predictions)

    print("\n========== RANDOM FOREST ==========")
    print(f"MAE:  {rf_mae:.2f}")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"R²:   {rf_r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, rf_predictions, alpha=0.3, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)

    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Random Forest: Actual vs Predicted Prices')

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'rf_plot.png'))
    plt.close()

    print("Plot saved: rf_plot.png")
    
# ====== Section: Model Comparison ======
    results = pd.DataFrame({'Model': ['Linear Regression', 'Random Forest'], 'MAE': [lr_mae, rf_mae], 'RMSE': [lr_rmse, rf_rmse], 'R2': [lr_r2, rf_r2]})

    print("\n========== MODEL COMPARISON ==========")
    print(results.to_string(index=False))

    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    plt.figure(figsize=(10, 6))

    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')

    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.tight_layout()

    plt.savefig(os.path.join(images_dir, 'feature_importance.png'))
    plt.close()

    print("Plot saved: feature_importance.png")

    print("\n========== SAMPLE PREDICTIONS ==========")

    for i in range(5):
        actual = y_test.iloc[i]
        predicted = rf_predictions[i]
        diff = abs(actual - predicted)
        print(f"House {i+1}: Actual = ${actual:,.0f}, Predicted = ${predicted:,.0f}, Error = ${diff:,.0f}")
    if rf_r2 > lr_r2:
        improvement = ((rf_r2 - lr_r2) / lr_r2) * 100
        print("\n========== CONCLUSION ==========")
        print(f"Random Forest outperforms Linear Regression.")
        print(f"R² improved from {lr_r2:.4f} to {rf_r2:.4f} ({improvement:.1f}% better)")
        print(f"MAE reduced from ${lr_mae:,.0f} to ${rf_mae:,.0f}")
    else:
        print("\n========== CONCLUSION ==========")
        print(f"Linear Regression outperforms Random Forest.")
        print(f"R² improved from {rf_r2:.4f} to {lr_r2:.4f}")
    
if __name__ == "__main__":
    main()
