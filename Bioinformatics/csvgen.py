import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle


# data = pd.read_csv("breast-cancer-wisconsin.data")

# data.columns = ["id","ClumpThick","UniSize","UniShape","MargAd","SingEpiCelSize","Bare Nuc", "BlandChr", "NormalNuc","Mito","Class"]
# data.to_csv("data.csv", index=None, header=True)

data = pd.read_csv("data.csv")

data.drop(['id'], inplace = True, axis = 1)
data.replace('?', -99999, inplace = True)

def retBin(x):
    if x == 4:
        return 1
    else:
        return 0

data["Class"] = data["Class"].map(retBin)
data["Class"] = data["Class"].map(lambda x: 1 if x == 4 else 0)

# Defining X and y (Features and Labels)
X = np.array(data.drop(["Class"], axis = 1))
y = np.array(data["Class"])

############# Training and Testing the Models ##############
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.1, random_state = 0)

# # SVC Classifier
Classifier = SVC(kernel = 'linear')
model = Classifier.fit(X_train, y_train)
accu = model.score(X_test, y_test)
print("Accuracy of SVC: ", accu)

# # Logistic Regression
Classifier = LogisticRegression(solver = 'liblinear')
model = Classifier.fit(X_train, y_train)
accu = model.score(X_test, y_test)
print("Accuracy of Logistic Regression : ", accu)

# Saving and Loading the Models #

#Save the model
pickle.dump(model, open("LogisticRegression.m", "wb"))

# Loading The model
loaded_model = pickle.load(open("LogisticRegression.m", "rb"))
accu = loaded_model.score(X_test, y_test)
print("Accuracy of Logistic Regression : ", accu)

###### Making Predictions  #######
classes = ["Benign", "Malignant"]
sample = np.array([[5,10,10,10,7,7,3,8,9]])
result = loaded_model.predict(sample)
print(classes[int(result)])


sample2 = np.array([[3,1,1,1,2,2,3,1,1]])
result = loaded_model.predict(sample2)
print(classes[int(result)])


sample3 = np.array([[8,10,10,8,7,10,9,7,1]])
result = loaded_model.predict(sample3)
print(classes[int(result)])































