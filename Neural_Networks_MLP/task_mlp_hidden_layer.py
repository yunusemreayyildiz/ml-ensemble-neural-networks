#Task 1.2 Multi Layer Perceptron
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)# to ignore convergence warnings.
# Effects of hidden layers

def plot_layer_effects():
    #Load digit dataset
    digits = load_digits()
    test_scores = []
    train_scores = []
    layer_sizes = range(1, 11)
    #70%-30% split of dataset randomly selected
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)
    for i in layer_sizes :
        # Apply MLP network with i hidden layer size 
        # define hidden layers as
        #2^j, 2^(j-1), ..., 2^1 for each j in i 
        hidden_Layers = tuple([2**j for j in range(i, 0, -1)])# 2^j, 2^(j-1), ..., 2^1(decreasing order)
        print(f"Layer number :{i} Hidden Layers: {hidden_Layers}")
        mlp = MLPClassifier(hidden_layer_sizes=hidden_Layers, max_iter=100, random_state=42)
        mlp.fit(X_train, y_train)
        train_scores.append( mlp.score(X_train, y_train))
        test_scores.append( mlp.score(X_test, y_test))
    # Plot score values as a function of hidden layer size
    plt.figure(figsize=(10, 6))
    plt.plot(layer_sizes, train_scores, marker='*', label='Train', color='black')
    plt.plot(layer_sizes, test_scores, marker='+', label='Test', color='red')
    plt.title('Train & Test scores as a function of hidden layer size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Accuracy Score')
    plt.legend()# takes the label for both of the lines and shows them in same plot
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_layer_effects()
