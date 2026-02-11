# Task 1.1: Perceptron Networks
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # necessary for 3D plotting
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
# 1.2 Visualization of decision boundary
def visualize_decision_boundry():
        # Generate a binary-class dataset for classification task.
        # for binary-class classification (n_classes=2)
        X, y = make_classification(
            n_samples=1000,#number of tuples
            n_features=3,# total number of features
            n_classes=2,# number of usable classes/features
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42 #Hitchhikers guide reference :)
        )
        #at least 500 tuples and three features of which two features are 
        # informative to the ground truth vector. 
        # Split the dataset into training and test sets 70%-30%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        #Model training
        classfier = Perceptron(max_iter=100, eta0=0.1, random_state=42)#edefining model and with eta0=0.1 defining learning rate of the model ,like if model did error take moderate steps sized at 10% of the error.'"
        #Apply single-layered perceptron network to fit the training data.
        classfier.fit(X_train, y_train)
        #plot testing objects and hyperplane 
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 7))# defining figure size
        axes = fig.add_subplot(111, projection='3d')#defining 3d axes (structure of figure)(defining only enviroment for 3d plot)
        axes.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], c='darkblue', label='Type 1', marker='*')#replace the test data points on defined environment
        axes.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], c='red', label='Type 2', marker='o')#
        # Draw Hyperplane
        w = classfier.coef_[0]# weights(w1,w2,w3)
        b = classfier.intercept_[0] # bias (constant)
        # Create grid to plot the hyperplane
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                             np.linspace(y_min, y_max, 20))
        # Hyperplane equation: w1*x1 + w2*x2 + w3*x3 + b = 0
        #solving for x3 (z axis) = -(w1*x + w2*x +b)/w3
        zz = (-w[0] * xx - w[1] * yy - b) / w[2] # define z axis based on hyperplane equation
        axes.plot_surface(xx, yy, zz, alpha=0.4, color='purple') # plotting hyperplane surface that separates two classes
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set_title('3D Decision Boundary of Perceptron Visualization')
        axes.legend()#using for drawing lejiant 
        plt.show()

if __name__ == "__main__":
    visualize_decision_boundry()

    