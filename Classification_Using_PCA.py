###################################################################################################################
# Compressing Feature Space For Classification Using Principal Component Analysis (PCA)
###################################################################################################################
"""
In this project we use Principal Component Analysis (PCA) to compress 100 unlabelled, sparse features into a more manageable number for classiying buyers of Ed Sheeran’s latest album.


Project Overview

Context
Our client is looking to promote Ed Sheeran’s new album - and want to be both targeted with their customer communications, and as efficient as possible with their \nmarketing budget.

As a proof-of-concept they would like us to build a classification model for customers who purchased Ed’s last album based upon a small sample of listening data they \nhave acquired for some of their customers at that time.

If we can do this successfully, they will look to purchase up-to-date listening data, apply the model, and use the predicted probabilities to promote to customers who \nare most likely to purchase.

The sample data is short but wide. It contains only 356 customers, but for each, columns that represent the percentage of historical listening time allocated to each of \n100 artists. On top of these, the 100 columns do not contain the artist in question, instead being labelled artist1, artist2 etc.

We will need to compress this data into something more manageable for classification!


Actions
We firstly needed to bring in the required data, both the historical listening sample, and the flag showing which customers purchased Ed Sheeran’s last album. We ensure we split our data a training set & a test set, for classification purposes. For PCA, we ensure that we scale the data so that all features exist on the same scale.

We then apply PCA without any specified number of components - which allows us to examine & plot the percentage of explained variance for every number of components. Based upon this we make a call to limit our dataset to the number of components that make up 75% of the variance of the initial feature set (rather than limiting to a specific number of components). We apply this rule to both our training set (using fit_transform) and our test set (using transform only)

With this new, compressed dataset, we apply a Random Forest Classifier to predict the sales of the album, and we assess the predictive performance!
"""



###############################################################################
#Import required Packages
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

###############################################################################
#Import Sample Data
###############################################################################

#Import
data_for_model = pd.read_csv(open("data/sample_data_pca.csv"))

#Drop unncessary columns
data_for_model.drop("user_id", axis=1, inplace= True)

#Shuffle data
data_for_model = shuffle(data_for_model, random_state= 42)

#Class Balance
data_for_model["purchased_album"].value_counts(normalize= True) #without normalize= True, it gives the count for number of 0's & 1's, with normalize= True, it gives percentage values

###############################################################################
#Deal with missing values
###############################################################################
data_for_model.isna().sum().sum()
data_for_model.dropna(how= "any", inplace= True)

###############################################################################
#Split Input & Output Variables
###############################################################################
X = data_for_model.drop(["purchased_album"], axis= 1)
y = data_for_model["purchased_album"]

###############################################################################
#Split Data into Train & Test 
###############################################################################

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42, stratify= y)

###############################################################################
#Feature Scaling
###############################################################################

scale_standard = StandardScaler()

X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)

###############################################################################
#Apply Principal Component Analysis - PCA
###############################################################################

#Instantiate & fit

pca = PCA(n_components = None, random_state = 42)
pca.fit(X_train)

#Extract the explained variance across components

explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()

###############################################################################
#Plot the explained variance across components
###############################################################################

#create list for number of components

num_vars_list = list(range(1,101))
plt.figure(figsize= (15,10))

#Plot the variance explained by each component

plt.subplot(2,1,1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("% Variance")
plt.tight_layout()

#Plot the cumulative variance

plt.subplot(2,1,2)
plt.plot(num_vars_list, explained_variance_cumulative)
plt.title("Cumulatiove Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()
plt.show()

###############################################################################
#Apply PCA with selected number of components
###############################################################################

pca = PCA(n_components = 0.75, random_state = 42)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.n_components_

###############################################################################
#Building classifier model
###############################################################################

clf = RandomForestClassifier(random_state= 42)
clf.fit(X_train, y_train)

###############################################################################
#Assess Model Accuracy
###############################################################################

y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

