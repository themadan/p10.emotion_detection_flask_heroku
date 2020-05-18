from src.features.tokenization import tokenization
from sklearn.naive_bayes import MultinomialNB

def Bayesian():


    X_test_counts,X_test_counts=tokenization()
    clf = MultinomialNB().fit(X_train_counts,y_train)
    predicted = clf.predict(X_test_counts)
    nb_clf_accuracy = np.mean(predicted == y_test) * 100
    print(nb_clf_accuracy)

if __name__ == "__main__":
    Bayesian()