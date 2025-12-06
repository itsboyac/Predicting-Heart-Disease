"""
Heart Disease Prediction Model Training Script

This script trains a K-Nearest Neighbors classifier for heart disease prediction
and saves the trained model and scaler for use in the API.
"""

import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_and_preprocess_data(filepath='heart.csv'):
    """Load and preprocess the heart disease dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y, list(X.columns)


def train_model(X, y):
    """Train KNN model with hyperparameter tuning."""
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning with GridSearchCV
    print("Training KNN model with GridSearchCV...")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return best_model, scaler


def save_model_and_scaler(model, scaler, feature_names):
    """Save the trained model and scaler to disk."""
    print("\nSaving model and scaler...")
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to model.pkl")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")
    
    # Save feature names for reference
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("Feature names saved to feature_names.pkl")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Heart Disease Prediction Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Save artifacts
    save_model_and_scaler(model, scaler, feature_names)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
