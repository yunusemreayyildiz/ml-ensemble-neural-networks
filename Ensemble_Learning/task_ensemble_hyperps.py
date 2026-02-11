from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)# to shows outputs clearly

def run_ensemble_hyperps():
    # Load the Breast Cancer dataset (a binary classification problem).
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    #List to store individual MLP estimators for the VotingClassifier.
    estimators_list = []
    # Loop to create 10 different MLP models (h=1 to h=10).
    for h in range(1, 11):
        #define the hidden layer structure: hidden_layers is a tuple of layer sizes.
        # It creates a structure where the layer sizes are powers of 2, decreasing from 2^h to 2^1 (e.g., for h=3: (8, 4, 2)).
        hidden_layers = tuple([2**i for i in range(h, 0, -1)])
        
        # Initialize the Multi-layer Perceptron (MLP) Classifier.
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers, 
            max_iter=1200, 
            random_state=42
        )
        # Train the individual MLP model.
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Parameter setting: l#{h} Accuracy: {acc}")
        
        estimators_list.append((f'mlp_{h}', mlp))



    ensemble_clf = VotingClassifier(estimators=estimators_list, voting='hard')
    # Since the base estimators are already fitted, fit() here primarily aggregates them.
    ensemble_clf.fit(X_train, y_train)
    # Evaluate the Ensemble model's performance on the test set.
    ensemble_acc = ensemble_clf.score(X_test, y_test)
    
    print(f"Ensemble Learning Accuracy: {ensemble_acc}")

if __name__ == "__main__":
    run_ensemble_hyperps()