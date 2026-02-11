# Task-2 Ensemble Learning
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.inspection import DecisionBoundaryDisplay

def run_boosting():
    # 1. Load moon dataset with deviation value of 0.2
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    
    # 2. Split (%70 Train, %30 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Base Estimator
    base_estimator = SGDClassifier(loss="log_loss",learning_rate="constant",random_state=42,eta0=0.01,max_iter=1000)
    
    # 4. AdaBoost Initialize and train the ensemble with 4 weak estimators
    boosting_clf = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=4,
        algorithm='SAMME', 
        random_state=42
    )
    boosting_clf.fit(X_train, y_train)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, learner in enumerate(boosting_clf.estimators_):
        ax = axes[i]
        # Plot the decision boundary of the current learner
        DecisionBoundaryDisplay.from_estimator(
            learner, X, #Use the whole dataset X for boundary plotting context.
            plot_method="contour",
            ax=ax,
            levels=[0.5],
            linestyles='--',
            colors='black',
            response_method="predict"
        )
        #the points to differenciate
        ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', marker='x', label='Class 0')
        ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', marker='o', label='Class 1')
        
        ax.set_title(f"Learner #{i+1}")
        
    plt.tight_layout()
    plt.savefig('BaseLearnerVisualization.pdf')#save the graphs as pdf file
    plt.show(block=False)
    plt.close("all")

if __name__ == "__main__":
    run_boosting()