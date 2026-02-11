from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
def run_ensemble_learners():
    #Load Breast Cancer Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Base Classifiers 
    clf1 = LogisticRegression( max_iter=1000, random_state=42)
    clf2 = DecisionTreeClassifier(random_state=42)
    clf3 = KNeighborsClassifier(n_neighbors=5)

    voting_clf = VotingClassifier(
        estimators=[('logistic', clf1), ('dt', clf2), ('KNN', clf3)],
        voting='hard'
    )
    # Define labels and list of models for iteration and evaluation.
    labels = ['learner #1 logistic regression', 'learner #2 (DT)', 'learner #3 (KNN)', 'ensemble learner']
    models = [clf1, clf2, clf3, voting_clf]

    print("Ensemble Learning Results (5-Fold CV):")

    #Loop through the models to print the accuracies
    for i in range(len(models)):
        #Calculate the 5-fold cross-validation scores for the current model.
        scores = cross_val_score(models[i], X, y, cv=5, scoring='accuracy')
        print(f"Accuracy obtained by {labels[i]} is: {scores.mean():.4f}")

if __name__ == "__main__":
    run_ensemble_learners()