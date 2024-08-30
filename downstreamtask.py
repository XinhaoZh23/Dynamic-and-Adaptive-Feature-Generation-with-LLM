from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
def train_classifier(X_train, y_train, model_name='DecisionTree', model_params=None):
    """
    Trains a classifier on the provided training data based on the specified model name.

    Parameters:
    X_train (DataFrame): Feature matrix for training.
    y_train (Series): Labels for training.
    model_name (str): Name of the model to train ('DecisionTree' or 'NeuralNetwork').

    Returns:
    A trained model (either DecisionTreeClassifier or a PyTorch model).
    """
    if model_name.lower() == 'decision_tree':
        # Initialize and train the Decision Tree Classifier
        clf = DecisionTreeClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name.lower() == 'random_forest':
        # Initialize and train the Random Forest Classifier
        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name.lower() == 'knn':
        # Initialize and train the K-Nearest Neighbors Classifier
        clf = KNeighborsClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name.lower() == 'mlp':
        # Initialize and train the Multi-Layer Perceptron Classifier
        clf = MLPClassifier(**model_params)
        clf.fit(X_train, y_train)
        return clf

    elif model_name == 'NeuralNetwork':
        # Convert DataFrame to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Define a simple neural network model
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, len(y_train.unique()))
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return model
    else:
        raise ValueError("Unsupported model name provided.")

def test_classifier(clf, X_test, y_test):
    """
    Tests a trained classifier on the provided test data and calculates accuracy, precision, recall, and F1 score.

    Parameters:
    clf (classifier): A trained classifier which can be DecisionTreeClassifier, RandomForest, KNN, MLP, or a PyTorch model.
    X_test (DataFrame): Feature matrix for testing.
    y_test (Series): Actual labels for testing.

    Returns:
    tuple: A tuple containing the predictions and the scores (accuracy, precision, recall, F1).
    """
    # Check if the classifier is a PyTorch model
    if isinstance(clf, torch.nn.Module):
        # Convert DataFrame to PyTorch tensors
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

        # Ensure model is in evaluation mode
        clf.eval()

        # Predict on the test data
        with torch.no_grad():
            outputs = clf(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.numpy()  # Convert predictions to numpy array for compatibility with sklearn metrics

    else:
        # Predict using a scikit-learn model
        y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # Macro averaging for multi-class classification
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Return all calculated metrics
    return y_pred, (accuracy, precision, recall, f1)

