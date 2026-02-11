# Task-2 Ensemble Learning
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler

# Different training sets 
#  Bagging & Pasting
def run_bagging():
    #2.1.1 Load digit dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Scale data for better MLP performance
    # MLP works better when data is standardized
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # split of dataset randomly selected 70-30%
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Define base MLP classifier
    base_mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8, 4, 2), #four hidden layers and 2 neurons in order 
        max_iter=2000, 
        random_state=42,
        solver='lbfgs', # solver for small datasets
        activation='tanh' # tanh helps with bottleneck layers
    )
    
    #Define Bagging Classifier with 8 estimators 
    bagging_clf = BaggingClassifier(
        estimator=base_mlp,
        n_estimators=8,
        max_samples=0.125, # %12.5 of training data(since there are 8 estimators)
        bootstrap=True,    
        random_state=42,
        n_jobs=-1   # use all cpu cores
    )
    
    #train the bagging classifier
    bagging_clf.fit(X_train, y_train)
    
    total_test_samples = len(y_test)
    
    print("\nIndividual Learner Performance:")
    #Calculate accuracy for each learner
    for i, learner in enumerate(bagging_clf.estimators_):
        y_pred_individual = learner.predict(X_test)
        correct_count = np.sum(y_pred_individual == y_test)
        print(f"{correct_count} out of {total_test_samples} instances are correctly classified by learner #{i+1}")

    # Calculate accuracy for bagging classifier
    y_pred_bagging = bagging_clf.predict(X_test)
    correct_count_bagging = np.sum(y_pred_bagging == y_test)
    
    print("-" * 50)
    print(f"{correct_count_bagging} out of {total_test_samples} instances are correctly classified by bagging")
    print(f"Bagging Accuracy: {correct_count_bagging / total_test_samples:.2%}")

if __name__ == "__main__":
    run_bagging()