# WSDM-Churn-Prediction
ML algorithm to predict whether a user will renew their subscription or not - Kaggle Challenge

modelcompare.py - program to run a comparision of several classifiers with the database. 

modelevaluate.py - program to predict whether the user will renew their subscription or not

Team members: Adithya Ganapathy, Srihari Murali and Nisshanthni Divakaran

Language Used: Python

Dataset Used: https://www.kaggle.com/lifinger/wsdm-kkbox/data

Steps to Compile:
1. Extract files.zip 
1. Open command prompt
2. Change directory to <../project>
3. Run the following commands
	python modelevaluate.py <filepath>
Example:python modelevaluate.py "C:/project/files"

For comparision of models:
1. Open command prompt
2. Change directory to <../project>
3. Run the following commands
	python modelcompare.py <trainingfile>
Example:python modelcompare.py "C:/project/files/trainfinal.csv" 
