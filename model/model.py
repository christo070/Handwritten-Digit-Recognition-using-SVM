import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Using dataset
train_dataset = pd.read_csv('model/mnist_train.csv')
test_dataset = pd.read_csv('model/mnist_test.csv')

## Separating the X and Y variable
y = train_dataset['label'].values.astype('int32')
## Dropping the variable 'label' from X variable
X = train_dataset.drop(columns = 'label').values.astype('float32')

## Normalization
X = X/255.0

# Scaling the features
X_scaled = scale(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, train_size = 0.8, random_state = 10)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

with open("model/model.pkl", 'wb') as file:  
    pickle.dump(classifier, file)