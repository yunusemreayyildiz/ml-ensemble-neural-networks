#Task 1.1 : Perceptron Networks 
import warnings # to ignore convergence warnings(icreasing readability of output)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
import time # for time measurement
from sklearn.datasets import make_classification # creator
import sklearn.model_selection as ms# splitter
from sklearn.linear_model import Perceptron# model
# Single-Layered percepton
# 1.1 Classification task 
# this function takes number of examples, number of features and epochs as input
# and returns average learning time and average error over 10 iterations
def run_experiment(n_of_examples,n_of_features,epochs):
    total_learning_time_ms = 0  #this variable will hold total learning time over 10 iterations
    total_error = 0 # this variable will hold total error over 10 iterations
    for i in range(10): 
        #1.1.1 data generation with using make_classification
        # for binary-class classification (n_classes=2)
        X, y = make_classification(
            n_samples=n_of_examples,
            n_features=n_of_features,
            n_classes=2,# number of usable classes/features
            n_informative=min(2, n_of_features), #handle case when n_of_features < 2
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42 
        )
        #1.1.2 data splitting with using train_test_split %70 - %30
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3 ,random_state=42) 
        #1.1.3 model creation with using Perceptron 
        model = Perceptron(max_iter=epochs, eta0=0.1, random_state=42) #defining model with moderate step size as 10%   
        #1.1.4 model training and time measurement
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        learning_time_ms = (end_time - start_time)*1000 #to convert s to ms 
        #print(f"Iteration {i+1}: Learning time: {learning_time_ms:.6f} ms")#I wrote this line to be sure about each iterations works well
        total_learning_time_ms += learning_time_ms

        #1.1.5 model testing and error calculation
        accuracy = model.score(X_test, y_test)
        #print(f"Iteration {i+1}: Accuracy: {accuracy}")
        error = error = 1 - accuracy
        total_error += error # accumulate error over iterations
    return total_learning_time_ms / 10, total_error / 10 # return average learning time and average error divide by 10


if __name__ == "__main__":
    scenarios100 =[  
        # (n_of_examples, n_of_features, epochs) 
        ('a',10000, 100, 100),
        ('b',10000, 1000, 100),
        ('c',100000, 100, 100),
        ('d',250000, 100, 100), 
        ]
    scenarios500 =[  
        ('e',10000, 100, 500),
        ('f',10000, 1000, 500),
        ('g',100000, 100, 500),
        ('h',250000, 100, 500), 
        ]        

    print("100 iterations Scenarios:")
    for scenario in scenarios100:
        name , n_of_examples , n_of_features, epochs = scenario # assigning values to variables
        avg_time, avg_error = run_experiment(n_of_examples,n_of_features,epochs)# run the function for various scenarios
        print(f"Scenario {name}: Avg Learning Time: {avg_time:.6f} ms, Avg Error: {avg_error:.4f}")
    print("500 iterations Scenarios:")
    for scenario in scenarios500:
        name , n_of_examples , n_of_features, epochs = scenario
        avg_time, avg_error = run_experiment(n_of_examples,n_of_features,epochs)
        print(f"Scenario {name}: Avg Learning Time: {avg_time:.6f} ms, Avg Error: {avg_error:.4f}")


