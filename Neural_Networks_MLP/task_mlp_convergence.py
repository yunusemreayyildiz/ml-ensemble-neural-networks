#Task-1.2 Multi-Layer Perceptron (MLP)
from sklearn.datasets import load_digits #Load digit dataset
import matplotlib.pyplot as plt # plotting
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier #we define the structure of neural network and its hyperparameters 
#2.1 Error convergence with MLP
def plot_error_convergence():
    #2.1.1 Load digit dataset
    digits = load_digits()
    X = digits.data 
    y = digits.target
    #2.1.2 70%-30% split of dataset randomly selected
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Create MLP model 
    #2.1.3 Apply MLP network with one hidden layer of 50 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, random_state=42, verbose=True,solver='sgd' )#used sgd as a solver to make plot more optimize  (sgd refers to stochastic gradient descent)
    # verbose=True parameter that prints loss at each iteration ,helps to detect any issues during training
    mlp.fit(X_train, y_train)# Train the model
    #2.1.4 Plot error values as a function of iteration
    plt.plot(mlp.loss_curve_, color = 'darkblue')
    plt.title('MLP Convergence: error vs Iterations')#title of the plot
    plt.xlabel('Iterations')
    plt.ylabel('error (loss)')
    plt.show()
if __name__ == "__main__":
    plot_error_convergence()