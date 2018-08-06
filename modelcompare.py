# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:26:10 2017

@authors: Adithya Ganapathy(axg172330)
          Sri Hari Murali(sxm179330)
          Nisshantni Divakaran(nxd171330)
"""

import gc; gc.enable()
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#Reading the file and 
filepath = sys.argv[1]
train = pd.read_csv(filepath)
cols = [c for c in train.columns if c not in ['is_churn','msno']]

#Test train split for creating training and validation file
train_X, test_X, train_y, test_y = train_test_split(train[cols], train['is_churn'], test_size=0.25)
scaler = RobustScaler()
scaler.fit(train_X)
X_train_pca = scaler.transform(train_X)
X_test_pca = scaler.transform(test_X)

#Randomforest Classifier
model_rfc = RandomForestClassifier(n_estimators = 50)
model_rfc.fit(X_train_pca,train_y)
predictions = model_rfc.predict(X_test_pca)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Random Forest", roc_auc))
randomforestaccuracy = accuracy_score(predictions,test_y)
randomforestprecision = precision_score(predictions, test_y)
randomforestrecall = recall_score(predictions, test_y)
randomforestf1 = f1_score(predictions, test_y)
print("Random forest Accuracy: %0.2f" % (randomforestaccuracy*100))
print("Random forest Precision: %0.2f" % (randomforestprecision))
print("Random forest Recall: %0.2f" % (randomforestrecall))
print("Random forest F1-Score: %0.2f" % (randomforestf1))

#NaiveBayes Classifier
model_nb = GaussianNB()
model_nb.fit(X_train_pca,train_y)
predictions = model_nb.predict(X_test_pca)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Naive Bayes", roc_auc))
nbaccuracy = accuracy_score(predictions, test_y)
nbprecision = precision_score(predictions, test_y)
nbrecall = recall_score(predictions, test_y)
nbf1 = f1_score(predictions, test_y)
print("-------------------------------------------------")
print("NaiveBayes Accuracy: %0.2f" % (nbaccuracy*100))
print("NaiveBayes Precision: %0.2f" % (nbprecision))
print("NaiveBayes Recall: %0.2f" % (nbrecall))
print("NaiveBayes F1-Score: %0.2f" % (nbf1))

#Logistic Regression Classifier
lr = LogisticRegression(penalty='l1', tol=0.1) 
lr.fit(X_train_pca,train_y)
predictions = lr.predict(X_test_pca)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Logistic Regression", roc_auc))
lraccuracy = accuracy_score(predictions, test_y)
lrprecision = precision_score(predictions, test_y)
lrrecall = recall_score(predictions, test_y)
lrf1 = f1_score(predictions, test_y)
print("-------------------------------------------------")
print("Logistic Regression Accuracy: %0.2f" % (lraccuracy*100))
print("Logistic Regression Precision: %0.2f" % (lrprecision))
print("Logistic Regression Recall: %0.2f" % (lrrecall))
print("Logistic Regression F1-Score: %0.2f" % (lrf1))
 
#KNeighbours Classifier
nbrs = KNeighborsClassifier(n_neighbors=25)
nbrs.fit(X_train_pca,train_y)
predictions = nbrs.predict(X_test_pca)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("K Neighbours", roc_auc))
nbrsaccuracy = accuracy_score(predictions, test_y)
nbrsprecision = precision_score(predictions, test_y)
nbrsrecall = recall_score(predictions, test_y)
nbrsf1 = f1_score(predictions, test_y)
print("-------------------------------------------------")
print("KNN Accuracy: %0.2f" % (nbrsaccuracy*100))
print("KNN Precision: %0.2f" % (nbrsprecision))
print("KNN Recall: %0.2f" % (nbrsrecall))
print("KNN F1-Score: %0.2f" % (nbrsf1))

#Bagging Classifier
bagg = BaggingClassifier(n_estimators=30)
bagg.fit(X_train_pca,train_y)
predictions = bagg.predict(X_test_pca)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Bagging", roc_auc))
baggaccuracy = accuracy_score(predictions, test_y)
baggprecision = precision_score(predictions, test_y)
baggrecall = recall_score(predictions, test_y)
baggf1 = f1_score(predictions, test_y)
print("-------------------------------------------------")
print("Bagging Accuracy: %0.2f" % (baggaccuracy*100))
print("Bagging Precision: %0.2f" % (baggprecision))
print("Bagging Recall: %0.2f" % (baggrecall))
print("Bagging F1-Score: %0.2f" % (baggf1))

#Adaboost Classifier
adaboost = AdaBoostClassifier(n_estimators=100)
adaboost.fit(X_train_pca,train_y)
predictions = adaboost.predict(X_test_pca)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Adaboost", roc_auc))
adaboostaccuracy = accuracy_score(predictions, test_y)
adaboostprecision = precision_score(predictions, test_y)
adaboostrecall = recall_score(predictions, test_y)
adaboostf1 = f1_score(predictions, test_y)
print("-------------------------------------------------")
print("Adabost Accuracy: %0.2f" % (adaboostaccuracy*100))
print("Adabost Precision: %0.2f" % (adaboostprecision))
print("Adabost Recall: %0.2f" % (adaboostrecall))
print("Adabost F1-Score: %0.2f" % (adaboostf1))

#Gradboost Classifier
gradboost = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80)
gradboost.fit(X_train_pca,train_y)
predictions = gradboost.predict(X_test_pca)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Gradient Boosting", roc_auc))
gradboostaccuracy = accuracy_score(predictions, test_y)
gradboostprecision = precision_score(predictions, test_y)
gradboostrecall = recall_score(predictions, test_y)
gradboostf1 = f1_score(predictions, test_y)
print("-------------------------------------------------")
print("Gradboost Accuracy: %0.2f" % (gradboostaccuracy*100))
print("Gradboost Precision: %0.2f" % (gradboostprecision))
print("Gradboost Recall: %0.2f" % (gradboostrecall))
print("Gradboost F1-Score: %0.2f" % (gradboostf1))

#Deep Learning Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,15,20,15,10,12,15,15,10), max_iter=500, solver='lbfgs', alpha=0.05, activation='logistic')
mlp.fit(X_train_pca,train_y)
predictions = mlp.predict(X_test_pca)
mlpaccuracy = accuracy_score(predictions, test_y)
fpr, tpr, thresholds = roc_curve(test_y,predictions)
roc_auc  = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Deep Learning", roc_auc))
mlpprecision = precision_score(predictions, test_y, average='macro')
mlprecall = recall_score(predictions, test_y, average='macro')
mlpf1 = f1_score(predictions, test_y, average='macro')
print("-------------------------------------------------")
print("Deep Learning Accuracy: %0.2f" % (mlpaccuracy*100))
print("Deep Learning Precision: %0.2f" % (mlpprecision))
print("Deep Learning Recall: %0.2f" % (mlprecall))
print("Deep Learning F1-Score: %0.2f" % (mlpf1))

#SVM
model_svm = SVC()
model_svm.fit(X_train_pca,train_y)
predictions = model_svm.predict(X_test_pca)
svmaccuracy = accuracy_score(predictions,test_y)
fpr, tpr, thresholds = roc_curve(test_y, predictions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("SVM", roc_auc))
svmprecision = precision_score(predictions, test_y)
svmrecall = recall_score(predictions, test_y)
svmf1 = f1_score(predictions, test_y)
print("-------------------------------------------------")
print("SVM Accuracy: %0.2f" % (svmaccuracy*100))
print("SVM Precision: %0.2f" % (svmprecision))
print("SVM Recall: %0.2f" % (svmrecall))
print("SVM F1-Score: %0.2f" % (svmf1))

#Plotting ROC Curve and Area under ROC 
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()
plt.clf()

results = {}
kfold = 5

#Classifier comparision using Kfold Cross Validation
results['RandomForest'] = cross_val_score(model_rfc, X_train_pca, train_y, cv=kfold).mean()
results['KNeighbors'] = cross_val_score(nbrs, X_train_pca, train_y, cv=kfold).mean()
results['LogisticRegression'] = cross_val_score(lr, X_train_pca, train_y, cv = kfold).mean()
results['GradientBoosting'] = cross_val_score(gradboost, X_train_pca, train_y, cv = kfold).mean()
results['Bagging'] = cross_val_score(bagg, X_train_pca, train_y, cv = kfold).mean()
results['Adaboost'] = cross_val_score(adaboost, X_train_pca, train_y, cv = kfold).mean()
results['DeepLearning'] = cross_val_score(mlp,X_train_pca,train_y,cv=kfold).mean()
results['NaiveBayes'] = cross_val_score(model_nb,X_train_pca,train_y,cv=kfold).mean()
results['SVM'] = cross_val_score(model_svm,X_train_pca,train_y,cv=kfold).mean()

plt.bar(range(len(results)), results.values(), align='center')
plt.xticks(range(len(results)), list(results.keys()), rotation='vertical')
plt.show()






