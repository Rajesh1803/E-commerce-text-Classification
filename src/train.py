from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
# from preprocess import preprocess_data
import logging

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model on the training data with regularization.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained Logistic Regression model
    """
    model = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)  # L2 regularization
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier on the training data.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained Random Forest Classifier
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def cross_validate_model(model, X, y):
    """
    Applies cross-validation to evaluate model performance.
    :param model: Trained model
    :param X: Features
    :param y: Labels
    :return: Cross-validation scores
    """
    cv_scores = cross_val_score(model, X, y, cv=5)
    logging.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return cv_scores

def grid_search_tuning(X_train, y_train):
    """
    Performs hyperparameter tuning for RandomForestClassifier using GridSearchCV.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Best tuned model from GridSearchCV
    """
    param_grid = {
        'n_estimators': [100, 200],
        
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters from GridSearchCV: {grid_search.best_params_}")
    return grid_search.best_estimator_

def save_model(model, vectorizer, lsa, model_dir):
    """
    Saves the trained model and vectorizer to the specified directory.
    :param model: Trained model
    :param vectorizer: Fitted TF-IDF vectorizer
    :param model_dir: Directory to save the model and vectorizer
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    joblib.dump(lsa, os.path.join(model_dir, 'lsa.pkl'))

