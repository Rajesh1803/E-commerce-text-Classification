import pandas as pd
import logging
from src.preprocess import preprocess_data
from src.train  import train_logistic_regression,grid_search_tuning,save_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Load and preprocess the data
    logging.info("Starting the preprocessing...")
    file_path = "Data/ecommerceDataset.csv"
    X_train, X_test, y_train, y_test, vectorizer, lsa = preprocess_data(file_path)
    logging.info("Preprocessing complete.")
    
    # Train Logistic Regression Model
    model_logreg = train_logistic_regression(X_train, y_train)
    logging.info("Logistic Regression Model trained.")
    
    # Train Random Forest Model with hyperparameter tuning
    best_rf_model = grid_search_tuning(X_train, y_train)
    logging.info("Best Random Forest Model found with GridSearchCV.")
    
    # Evaluate both models
    y_pred_logreg = model_logreg.predict(X_test)
    y_pred_rf = best_rf_model.predict(X_test)

    logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    # Compare models and save the best model
    if logreg_accuracy > rf_accuracy:
        logging.info("Logistic Regression is the best model. Saving it.")
        save_model(model_logreg, vectorizer, lsa, 'models')
    else:
        logging.info("Random Forest is the best model. Saving it.")
        save_model(best_rf_model, vectorizer, lsa, 'models')

    # Print classification reports for both models
    print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
