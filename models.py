from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR


def LR (X_train, Y_train):
	model = LogisticRegression()
	model.fit(X_train, Y_train)
	print accuracy_score

def 