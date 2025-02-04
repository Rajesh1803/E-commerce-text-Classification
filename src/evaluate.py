from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from preprocess import preprocess_data

import logging

def evaluate_model(model, vectorizer, X_test, y_test):
    """
    Evaluates the model using classification report and confusion matrix.
    :param model: Trained model
    :param vectorizer: TF-IDF vectorizer
    :param X_test: Test features
    :param y_test: Test labels
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=vectorizer.get_feature_names_out(), yticklabels=vectorizer.get_feature_names_out())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    # Load the model and vectorizer
    model = joblib.load('models/model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')

    logging.info("Training the model...")
    X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled, vectorizer = preprocess_data("../Data/ecommerceDataset.csv")
    logging.info("Training the model...")
    # Train the model
    # Evaluate the model
    evaluate_model(model, vectorizer, X_test, y_test)

if __name__ == "__main__":
    main()
