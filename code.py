# Imported the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore all warning messages that may arise during the execution
import warnings
warnings.filterwarnings("ignore")

# To load the dataset in dataframe
df = pd.read_csv("breast-cancer.csv")

# Displays the first 5 rows of the dataset, which can provide information such as the column names or the pattern of the data
df.head()

# Displays randomly selected number of rows from the dataset
df.sample(7)

# Specifies the shape (number of rows and columns), In this dataset there are 569 rows and 32 columns
df.shape

# Provides the statistical description of only numeric column in the dataset
df.describe()

# Provides information such as data-type of the column and checks for null values
df.info()

# It provides the unique number of data points present in each column
df.nunique()

# Another way to check the total number of null values in the dataset
df.isna().sum()

# The heatmap provides the correlation between the columns
# Each column is highly correlated with itself
sns.heatmap(df.corr())

# Displays the corelation with  a value for better understanding
df.corr()

# Droped the ID column as it does not add value to the dataset and is not relevant to the context
df.drop(["id"],axis=1,inplace=True)

# The ID column is successfully dropped amd displayed the remaining column
df.head()

df["diagnosis"].value_counts()

# Countplot, which represents the number of malignant and benign tumor count present in the dataset of breast cancer
# M - Malignant & B - Benign
# people have more benign tumor than malignant tumor.
sns.countplot(data=df,x="diagnosis")

from sklearn.preprocessing import LabelEncoder
# Label Encoder is used to encode categorical column to numerical column.
le = LabelEncoder()
df["diagnosis"]=le.fit_transform(df["diagnosis"])

# this shows that the column diagnosis is successfully encoded to numeric value, in the form of 0 and 1
df.head()

# Now X is independent column, Y is the dependent column or (target) column.
# The X value consist of various features related to breast cancer, which will help to train our model
# The Y column is our target column(diagnosis), which will help to classify into two dictinct categories.
x = df.iloc[:,1:].values
y = df.iloc[:,0].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Next for training the model, fitted and transformed the training data, and transformed the testing data.
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# Imported the required libraries to create a aritificial neural network,
# This network will allow to create the input, hidden and output layer of the network.
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Initialize the neural network
model = Sequential()

# Input layer with 20 neurons and relu activation function
model.add(Dense(units=20, activation="relu"))

# Hidden layer with 15 neurons and relu activation function
model.add(Dense(units=15, activation="relu"))

# Output layer consist of only 1 neuron and sigmoid activation function as it is binary classification
model.add(Dense(units=1, activation = "sigmoid"))


# Compiled the model with adam optimizer and displayed the loss
model.compile(optimizer="adam",loss = "binary_crossentropy")

# Neural network training
model.fit(xtrain,ytrain,epochs=200,validation_data=(xtest,ytest))

# history specifies the loss and validation value of our neural network
lossdf = pd.DataFrame(model.history.history)
lossdf.head()
lossdf.tail()
lossdf.plot()

#For Early Stopping
model = Sequential()
model.add(Dense(units=20, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=1, activation = "sigmoid"))
model.compile(optimizer="adam",loss = "binary_crossentropy")

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor = "val_loss",mode="min",verbose=1,patience=25)
model.fit(xtrain,ytrain,epochs=200,validation_data=(xtest,ytest),callbacks=[earlystopping])

lossdf = pd.DataFrame(model.history.history)
lossdf.plot()

#For Dropout
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(Dense(units=20, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=15, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation = "sigmoid"))
model.compile(optimizer="adam",loss = "binary_crossentropy")

earlystopping = EarlyStopping(monitor = "val_loss",mode="min",verbose=1,patience=25)
model.fit(xtrain,ytrain,epochs=200,validation_data=(xtest,ytest),callbacks=[earlystopping])

lossdf = pd.DataFrame(model.history.history)
lossdf.plot()

# After completion of the training, predicted the output i.e the target variable on the testing dataset.
ypred = model.predict(xtest)
ypred=model.predict(xtest)
ypred = ypred > 0.5

from sklearn.metrics import classification_report
# Printed the classification report, which has specified the overall accuracy of the model, also the accuracy of diagnosis(0 & 1)
# The overall accuracy of model is 98%, the accuracy to predict only Malignant(1) tumor is 98%, 
# and to predict the Benign(0) tumor is 99%.
print(classification_report(ytest,ypred))

from sklearn.metrics import confusion_matrix
# A confusion matrix is a tabular representation that shows,
# the count of correct and incorrect predictions made by classification model

# to create the confusion matrix, we need to pass the value for prediction, so
# we need to pass the testing data and the predicted data of the targeted column.
conf_matrix = confusion_matrix(ytest,ypred, labels=[0,1])
conf_matrix
# array([[72,  0], this it confusion matrix I got.
#       [ 2, 40]]
# this specifies that there are 72 true positive, 40 true negative and 2 false positive, 0 false negative.

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


