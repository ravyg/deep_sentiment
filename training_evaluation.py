## Ricardo A. Calix, 2016 
## getting started with scikit-learn
#######################################################

#imports

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
import warnings 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

######################################
## set parameters
warnings.filterwarnings("ignore", category=DeprecationWarning) 



#######################################
## load problem data
## find a csv file and load it here


data_X = []
data_y = []

csvfile= list(csv.reader(open ('./feature_vector.csv', 'rU'), delimiter =','))

for data in csvfile:
   data_y.append(data[8])
   data_X.append(data[1:7])
    
le =LabelEncoder()
y=le.fit_transform(data_y)


X = data_X
# print X

#######################################
##load iris data for testing purposes


#######################################
## load breast cancer data



#######################################
## plotting function

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z,alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]               
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='test_set')
    
###################################################

def print_stats_10_fold_crossvalidation(algo_name, model, X_train, y_train ):
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    kfold = StratifiedKFold(y=y_train,
                        n_folds = 10, # number of folds
                        random_state=1) 
    print "----------------------------------------------"
    print "Start of 10 fold crossvalidation results"
    print "the algorithm is: ", algo_name
    #################################
    #roc
    fig = plt.figure(figsize=(7,5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    ################################
    scores = []
    f_scores = []
    precisionlst = []
    Recall_lst= []


    for k, (train, test) in enumerate(kfold):
        model.fit(X_train[train], y_train[train])
        y_pred = model.predict(X_train[test])
        print "------------------------------------------"

        ########################
        #roc
        probas = model.predict_proba(X_train[test])
        #pos_label in the roc_curve function is very important. it is the value
        #of your classes such as 1 or 2, for versicolor or setosa
        fpr, tpr, thresholds = roc_curve(y_train[test],probas[:,1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, 
                 label='ROC fold %d (area = %0.2f)' % (k+1, roc_auc))

        ########################
        ## print results
        print('Accuracy: %.2f' % accuracy_score(y_true=y_train[test], y_pred=y_pred))
        confmat = confusion_matrix(y_true=y_train[test], y_pred=y_pred)
        print "confusion matrix"
        print(confmat)
        print('Precision: %.3f' % precision_score(y_true=y_train[test], y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=y_train[test], y_pred=y_pred))
        f_score = f1_score(y_true=y_train[test], y_pred=y_pred)
        print('F1-measure: %.3f' % f_score)
        f_scores.append(f_score)
        precisionlst.append(precision_score(y_true=y_train[test], y_pred=y_pred))
        Recall_lst.append(recall_score(y_true=y_train[test], y_pred=y_pred))


        score = model.score(X_train[test], y_train[test])
        scores.append(score)
        print('fold : %s, Accuracy: %.3f' % (k+1, score))
    ######################################
    #roc
    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plot_ROC_curve(plt, mean_fpr, mean_tpr, mean_auc )
    ######################################
    print('overall accuracy: %.3f and +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('overall f1_score: %.3f' % np.mean(f_scores))
    print('overall recall: %.3f' % np.mean(Recall_lst))
    print('overall precsion: %.3f' % np.mean(precisionlst))



##################################################
## plot ROC curve: used to plot a graoh in terms of false positive and true positive as axis. meaning that sometimes it predicts something which is true and sometimes it predicts which is actually true. 

def plot_ROC_curve(plt, mean_fpr, mean_tpr, mean_auc ):
    #fig = plt.figure(figsize=(7,5))
    plt.plot( [0,1],
              [0,1],
              linestyle='--',
              color=(0.6, 0.6, 0.6),
              label='random guessing')
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot( [0,0,1],
              [0,1,1],
              lw=2,
              linestyle=':',
              color='black',
              label='perfect performance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characterstics')
    plt.legend(loc="lower right").set_visible(True)
    plt.show()



###################################################
## plot 2d graphs

def plot_2d_graph_model(model,start_idx_test, end_idx_test, X_train_std, X_test_std, y_train, y_test ):
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test)) 
    # first 70% is train, last 30% is test
    #so in y_combined from 1 to 941 is train data. from 942 to 1345 is test
    plot_decision_regions(X=X_combined_std, 
                      y=y_combined,
                      classifier=model,
                      test_idx=range(start_idx_test,end_idx_test)) #942,1344
    plt.xlabel('cnt')
    plt.ylabel('sum')
    plt.legend(loc='lower left')
    plt.show()

###################################################
## print stats train % and test percentage (i.e. 70% train
## and 30% test
def print_stats_percentage_train_test(algorithm_name, y_test, y_pred):
    print "algorithm name: ", algorithm_name
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print "Printing Confusion Matrix"
    print confmat
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    # Get Recall
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    # F-measure.
    print('F-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))




###################################################

def knn_rc(X_train_std,y_train,x_test_std,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
    knn.fit(X_train_std,y_train)
    y_pred =knn.predict(X_test_std)
    # print_stats_percentage_train_test("Knn",y_test,y_pred)
    print_stats_10_fold_crossvalidation("Knn",knn, X_train_std,y_train)
   

#######################################
## logistic regression

def logistic_regression_rc(X_train_std, y_train, X_test_std,y_test):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1000, random_state=0)
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    # print_stats_percentage_train_test("Logistic Regression", y_test, y_pred) 
    print_stats_10_fold_crossvalidation("Logistic Regression",lr,X_train_std,y_train)
    # plot_2d_graph_model(lr,95, 145, X_train_std, X_test_std, y_train, y_test)
    # lr_result = lr.predict_proba(X_test_std[0,:])
    # print "lr confidence"
    # print lr_result
    # y_pred = lr.predict(X_test_std[0,:])
    # print y_pred


#####################################################
## random forest



#######################################
def svm_rc(X_train_std, y_train, X_test_std,y_test):
    from sklearn.svm import SVC 
    svm=SVC(kernel='rbf',random_state=0,gamma=0.10,C=10,probability=True, cache_size=7000)
    svm.fit(X_train_std,y_train)
    y_pred=svm.predict(X_test_std)
    print_stats_percentage_train_test("SVM", y_test, y_pred) 
    print '***********'
    #print_stats_10_fold_crossvalidation("SVM",svm,X_train_std,y_train)
    #plot_2d_graph_model(svm,95, 145, X_train_std, X_test_std, y_train, y_test)

#######################################
# A perceptron
def simple_perceptron_rc(x_train_std, y_train, x_test_std, y_test):
    from sklearn.linear_model import Perceptron
    # providing optimal parameters for fi tuning algorithm.
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print_stats_percentage_train_test("simple perceptron", y_test, y_pred) 
    print_stats_10_fold_crossvalidation("SVM", y_test, y_pred) 

    # print_stats_10_fold_crossvalidation("Perceptron",ppn,X_train_std,y_train)
 



#######################################
## decision trees
## prints tree graph as well
def decision_trees_rc(X_train,y_train,X_test,y_test):
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='entropy',max_depth=30,random_state=0)
    tree.fit(X_train_std,y_train)
    y_pred = tree.predict(X_test_std)
    print_stats_10_fold_crossvalidation("Decision",tree,X_train_std,y_train)
    # print_stats_percentage_train_test("Decision", y_test, y_pred) 



###################################################
## create train and test sets, or put all data in train sets
## for k-fold cross validation
## also perform feature scaling


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Feature Scaling.
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#######################################
## does not plot roc curve


#######################################
## ML_MAIN()
# simple_perceptron_rc(X_train_std, y_train, X_test_std, y_test)
# knn_rc(X_train_std,y_train,X_test_std,y_test)
decision_trees_rc(X_train,y_train,X_test,y_test)
# logistic_regression_rc(X_train_std,y_train,X_test_std,y_test)
# svm_rc(X_train_std, y_train, X_test_std,y_test)


#######################################

print "<<<<<<DONE>>>>>>>"

