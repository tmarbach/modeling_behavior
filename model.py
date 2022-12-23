"""
File Overview
--------------
Contains all modeling stratgies to be used with Acceleration Data
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import pickle


def adaboost_classifier(X_train, X_test, y_train, y_test, classes_names, save_flag):
    """
    desc
        Run the AdaBoost Classification algorithm against the given test and train data.
    params
        X_train     - samples to train on
        X_test      - samples to test on
        y_train     - ground truth of training data
        y_test      - ground truth of test data
        parameters  - parameters for the SVM model, default it given
    return
        report  -   includes accuracy, percision per-class, 
                    recall per-class, macro average and weigted average
    """
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth =1),
     n_estimators=100,
     algorithm = "SAMME.R",
     learning_rate =0.5)
    ada_clf.fit(X_train, y_train)
    if save_flag:
        filename = 'finalized_ada_model.sav'
        pickle.dump(ada_clf, open(filename, 'wb'))
    y_pred=ada_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=classes_names,
        output_dict=True)

    return report, y_pred, ada_clf.classes_


def random_forest(X_train, X_test, y_train, y_test, classes_names, save_flag):
    """
    desc
        Run the Random Forest algorithm against the given test and train data.
    params
        X_train     - samples to train on
        X_test      - samples to test on
        y_train     - ground truth of training data
        y_test      - ground truth of test data
        parameters  - parameters for the SVM model, default it given
    return
        report  -   includes accuracy, percision per-class, 
                    recall per-class, macro average and weigted average
    """
    rf_clf=RandomForestClassifier(
        n_estimators=1000,
        max_leaf_nodes=len(classes_names),
        n_jobs=-1)

    rf_clf.fit(X_train, y_train)
    if save_flag:
        filename = 'finalized_rf_model.sav'
        pickle.dump(rf_clf, open(filename, 'wb'))
    y_pred=rf_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=classes_names,
        output_dict=True)

    return report, y_pred, rf_clf.classes_


def naive_bayes(X_train, X_test, y_train, y_test, classes_names, save_flag):
    """
    desc
        Run the Naive Bayes algorithm against the given test and train data.
    params
        X_train     - samples to train on
        X_test      - samples to test on
        y_train     - ground truth of training data
        y_test      - ground truth of test data
        parameters  - parameters for the SVM model, default it given
    return
        report  -   includes accuracy, percision per-class, 
                    recall per-class, macro average and weigted average
    """
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nb_clf = ComplementNB()
    nb_clf.fit(X_train, y_train)
    if save_flag:
        filename = 'finalized_nb_model.sav'
        pickle.dump(nb_clf, open(filename, 'wb'))
    y_pred = nb_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=classes_names,
        output_dict=True)

    return report, y_pred, nb_clf.classes_


def svm(X_train, X_test, y_train, y_test, classes_names, save_flag):
    """
    desc
        Run the Support Vector Machine algorithm against the given test and train data.
    params
        X_train     - samples to train on
        X_test      - samples to test on
        y_train     - ground truth of training data
        y_test      - ground truth of test data
        parameters  - parameters for the SVM model, default it given
    return
        report  -   includes accuracy, percision per-class, 
                    recall per-class, macro average and weigted average
    """
    svm_clf = Pipeline([
            ("scalar", StandardScaler()),
            ("linear_svc", 
            SVC(kernel = "poly",
                degree = 3,
                C = 5)),
    ])
    svm_clf.fit(X_train, y_train)
    if save_flag:
        filename = 'finalized_svm_model.sav'
        pickle.dump(svm_clf, open(filename, 'wb'))
    y_pred = svm_clf.predict(X_test)
    report = classification_report(
        y_test,
        y_pred, 
        target_names=classes_names,
        output_dict=True)

    return report, y_pred, svm_clf.classes_


def ensemble_SVM(X_train, X_test, y_train, y_test, classes_names,save_flag):
    # define the base models
    models = list()
    models.append(('svm1', SVC(probability=False, kernel='poly', degree=1)))
    models.append(('svm2', SVC(probability=False, kernel='poly', degree=2)))
    models.append(('svm3', SVC(probability=False, kernel='poly', degree=3)))
    models.append(('svm4', SVC(probability=False, kernel='poly', degree=4)))
    models.append(('svm5', SVC(probability=False, kernel='poly', degree=5)))
    # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='hard')
    fitted_ensemble = ensemble.fit(X_train, y_train)
    if save_flag:
        filename = 'finalized_ensemble_model.sav'
        pickle.dump(fitted_ensemble, open(filename, 'wb'))
    y_pred = fitted_ensemble.predict(X_test)
    report = classification_report(
        y_test,
        y_pred, 
        target_names=classes_names,
        output_dict=True)

    return report, y_pred, fitted_ensemble.classes_
