#!/usr/bin/env python
# coding: utf-8

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore')

from sklearn.utils import resample
import pandas as pd
import numpy
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

import pandas_profiling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import graphviz
from sklearn import tree

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_colwidth', -1)

# ################################################################

# Instantiate each classifier to be used.
RandForest = RandomForestClassifier()  # Create Random Forest estimator object
GradBoost = GradientBoostingClassifier()
xgBoost = XGBClassifier()
DecisionTree = tree.DecisionTreeClassifier()
LogisticRegression = LogisticRegression()
LDAnalysis = LinearDiscriminantAnalysis()
MLP = MLPClassifier()
SGD = SGDClassifier(max_iter=1000, tol=None)
NaiveBayes = GaussianNB()  # Create Naive Bayes estimator object
SupportVectorMachine = svm.SVC()  # Create SVC estimator object
ExtraTrees = ExtraTreesClassifier()
adaBoost = AdaBoostClassifier()
QDAnalysis = QuadraticDiscriminantAnalysis()

classifier_list = [RandForest, GradBoost, xgBoost, DecisionTree, LDAnalysis, MLP,
                   SGD, NaiveBayes, SupportVectorMachine, ExtraTrees, adaBoost, QDAnalysis, LogisticRegression]

# Initialize resultant dataframe
columns = ['Classifier', 'Parameters', 'Features with Importance', 'Accuracy', 'AUC', 'F1-Score',
           'Precision', 'Recall', 'Sensitivity', 'Specificity', 'Threshold',
           'True Positive', 'True Negative', 'False Negative', 'False Positive', 'n', 'Random State']
df = pd.DataFrame(columns=columns)


# ################################################################

def customRound(threshold: float, unroundedArray):
    # TODO: Error handling, return None, hints.
    """
    Round array elements with custom threshhold
    :param<threshhold>: Rounding threshhold.
    :param<unroundedArray>: Array of elements to be rounded.

    :return: On success rounded array, else None.

    """

    for idx, nmbr in enumerate(unroundedArray):
        if nmbr > 1 or nmbr < 0:
            print('ERROR: Out of bounds (0,1)')
        elif nmbr < threshold:
            unroundedArray[idx] = int(numpy.floor(nmbr))
        elif nmbr >= threshold:
            unroundedArray[idx] = int(numpy.ceil(nmbr))
    roundedArray = unroundedArray
    return (roundedArray)


# ################################################################

def get_variable_importance(x_data, clf, thresh: float):
    # TODO: Error handling, return None, hints.
    """
        Get variable importance ranking
        :param<x_data>: Pandas dataframe of features.
        :param<clf>: Classification object after fit has been performed.
        :param<thresh>: Importance threshhold determines which features meet the importance requirement.

        :return: On success dictionary of feature importance ranks, else empty dictionary.

    """

    feature_names = x_data.columns.values
    importance = clf.feature_importances_
    importance_dict = dict(zip(feature_names, importance))
    # new_dict = { k : v for k,v in old_dict.iteritems() if v} for dictionary comprehension
    features = {feature: rank for feature, rank in importance_dict.items() if rank >= thresh}
    return (features)


# ################################################################

def get_pandas_profiling(data: str):
    # TODO: Error handling, return None.
    """
            Get variable importance ranking
            :param<data>: Pandas data frame of features.

            :return: On success list of rejected variables, else None.

    """

    DF = data
    df = pd.read_csv(DF, delimiter=',')
    pfr_all = pandas_profiling.ProfileReport(df)
    rejected_variables = pfr_all.get_rejected_variables(threshold=0.9)

    return (rejected_variables)


# ################################################################


# '''Produce Decision Tree Diagram'''
# dot_data = tree.export_graphviz(clf, feature_names=('gross_sales','ex_sales'), out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("class_2_data_tree_diagram")

# ################################################################

# def save_model(model, model_name: str):
#     # TODO: Test this and parameterize the output path.
#     """
#     Save model fitting data for later use.
#     :param<clf>: Classification object.

#     :return: On success fitting data is saved to a pickle file, else None.

#     """
#     file_name = '/Users/stevenpurcell/Documents/Data/Pickled_models/finalized_model.sav'
#     pickle.dump(model, open(file_name, 'wb'))


# ################################################################

def normalize(X):
    X_normalized = preprocessing.normalize(X)
    return X_normalized


# ################################################################

# def make_figure(classifier, fig, fig_outpath):
#     dot_data = tree.export_graphviz(classifier, out_file=None)
#     graph = graphviz.Source(dot_data)
#     graph.render(fig_outpath + "tree_diagram" + str(fig))


# ################################################################

def grid_search_wrapper(clf, x_data, y_data, param_grid, scorers, refit_score='accuracy_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    X_train, X_cv, y_train, y_cv = train_test_split(x_data, y_data, test_size=.20,
                                                    random_state=numpy.random.randint(200))
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                               cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid_search.predict(X_cv.values)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    return grid_search


# ################################################################

def randomSearchCv(clf, X_train, y_train):
    params = clf.get_params()
    # pprint(params)

    # Number of trees in random forest
    n_estimators = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in numpy.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    param_keys = ['n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                  'bootstrap']
    param_values = [n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap]
    rand_dict = dict(zip(param_keys, param_values))
    # pprint(rand_dict)

    random_grid = {}

    for param_key in param_keys:
        if param_key in params:
            random_grid[param_key] = rand_dict[param_key]

    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=numpy.random.randint(200), n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    return rf_random.best_params_


# ################################################################

def predictions(probabilities, data):
    df_out = pd.merge(data, probabilities, how='left', left_index=True, right_index=True)
    return df_out


# ################################################################

def upsample_minority_class(data, random_state: int):
    resample_feature = str(data.columns[-1])
    y_data = data.iloc[:, -1]

    sample_count = y_data.value_counts()
    majority_class_n = max(sample_count.values)
    minority_class = list(sample_count.values).index(min(sample_count.values))
    majority_class = list(sample_count.values).index(max(sample_count.values))
    df_majority = data[data[resample_feature] == majority_class]
    # Upsample minority class
    df_minority_upsampled = resample(data[data[resample_feature] == minority_class],
                                     replace=True,  # sample with replacement
                                     n_samples=majority_class_n,  # to match majority class
                                     random_state=random_state)  # reproducible results

    # Combine majority class with upsampled minority class
    data = pd.concat([df_majority, df_minority_upsampled])

    return data


# ################################################################

def train_and_fit(clf, n: int, x_data, y_data, random_state, threshold=.5):
    # TODO: Error handling, return None.
    """
                Train and fit classification model
                :param<clf>: Classification object.
                :param<n>: Run the model n times and average the metrics.
                :param<x_data>: Pandas dataframe of features.
                :param<y_data>: Pandas dataframe of the y to be predicted.

                :return: On success dictionary of performance metrics and fitting data, else None.

    """
    scores, mean_auc, mean_tpr, mean_spec, mean_prec, mean_rec, mean_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cm0 = numpy.zeros((2, 2))
    # params = numpy.logspace(-6, -1, 10)
    # clf = GridSearchCV(estimator=clf)
    for i in range(n):
        X_train, X_cv, y_train, y_cv = train_test_split(x_data, y_data, test_size=.20, random_state=random_state)
        fit_data = clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_cv)[:, 1]
        preds = customRound(threshold, preds)

        '''Calculate performance metrics'''
        scores += metrics.accuracy_score(y_cv, preds)
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        mean_auc += roc_auc
        cm = metrics.confusion_matrix(y_cv, preds)
        mean_tpr += float(cm[0][0]) / numpy.sum(cm[0])
        cm0 += cm
        prec, rec, _, _ = metrics.precision_recall_fscore_support(y_cv, preds, pos_label=1,
                                                                  average='binary')
        mean_prec += numpy.mean(prec)
        mean_rec += numpy.mean(rec)
        mean_spec += rec
        mean_f1 += 2 * numpy.mean(prec) * numpy.mean(rec) / (numpy.mean(prec) + numpy.mean(rec))

    '''Calculate mean performance metrics'''
    mean_accuracy = scores / n
    auc = mean_auc / n
    mean_sensitivity = mean_tpr / n
    mean_specificity = mean_spec / n
    mean_precision = mean_prec / n
    mean_recall = mean_rec / n
    f1 = mean_f1 / n

    metrics_dict = {'Accuracy': float(mean_accuracy),
                    'AUC': float(auc),
                    'Sensitivity': float(mean_sensitivity),
                    'Specificity': float(mean_specificity),
                    'Precision': float(mean_precision),
                    'Recall': float(mean_recall),
                    'F1-Score': float(f1),
                    'True Positive': int(cm0[1, 1]),
                    'True Negative': int(cm0[0, 0]),
                    'False Positive': int(cm0[0, 1]),
                    'False Negative': int(cm0[1, 0])}

    return (metrics_dict, fit_data, random_state)


# ################################################################

def build_report(df, classifier, x_data, y_data, random_state=42, threshold=.5, n=1,
                 feature_importance_threshold=.001):
    try:
        model_metrics, model_fit, random_state = train_and_fit(classifier, n, x_data, y_data, random_state,
                                                               threshold)
        feature_importance = get_variable_importance(x_data, model_fit, feature_importance_threshold)

        x_columns = list(feature_importance.keys())
        x_data = data[x_columns]
        feature_importance_list = list(feature_importance.items())
        model_metrics, model_fit, random_state = train_and_fit(classifier, n, x_data, y_data, random_state,
                                                               threshold)

        model_metrics['Features with Importance'] = str(feature_importance_list)
        model_metrics['Classifier'] = str(classifier).split('(')[0]
        model_metrics['Parameters'] = (str(classifier).split('(')[1]).replace('\n', ' ')
        model_metrics['Threshold'] = threshold
        model_metrics['n'] = n
        model_metrics['Random State'] = random_state

        s = pd.DataFrame(model_metrics, index=[0], columns=columns)
        df = pd.concat([s, df])
    except AttributeError:
        pass
    return df

# ################################################################
# Calling the functions

# Initialize starting conditions
random_state = 42
n = 1

data = '~/ImageVolumes/Jupyter/data/reduced_classification_features.csv'
data = pd.read_csv(data)

data = upsample_minority_class(data, random_state)
x_data = data.iloc[:, 0:-1]
y_data = data.iloc[:, -1]

for classifier in classifier_list:
    for threshold in numpy.arange(.01, 1, .01):
        df = build_report(df, classifier, x_data, y_data, random_state, threshold, n)


df.sort_values('F1-Score', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
df.reset_index(level=None, drop=True, inplace=True, col_level=0, col_fill='')
result = df.to_csv('output.csv')
print('fin')