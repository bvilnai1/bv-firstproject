# -*- coding: utf-8 -*-
"""
Module to simulate a neuroscience experiment. 


- [ ] TODO
"""
#%% Intialize
#Import sk library data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 


#%%
#Import data that is being testes
import pandas as pd
df = pd.read_csv(r'/Users/bryannavilnaigre/cfsc_2023/cfsc2023/examples/video_behavior_analysis.csv')
print(df) 

test_data = pd.read_csv(r'/Users/bryannavilnaigre/cfsc_2023/cfsc2023/examples/video_behavior_test_features.csv')
 
#%%
#Plot data 


#from matplotlib import pyplot as plt
#df.plt.figure()
#plt.hist(X,)
#plt.axvline(X_mean, color='r')
#plt.plot(X, np.repeat(0.75,X.shape), 'd', markersize=10, color=[0,0,0.3])
#plt.xlabel('X Values')
#plt.ylabel('Counts')
#plt.title(f'Histogram of {num_samples} samples (diamonds) and mean (vertical line)')
#plt.xlim( [xlo-0.5, xhi+0.5] )

#%%
#Fit data 

#features =x, label =y
features = df.iloc[:,1:6].values
label = df.iloc[:,6].values

features_train, features_test, label_train, label_test = train_test_split(features,label, test_size = 0.07)


scaler = StandardScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)


#%%
# Use #KNN to classify data 

classifier = KNeighborsClassifier(n_neighbors = 7)
label_classifier = classifier.fit(features_train, label_train) 


#%%

#Pedict label 

label_predict = classifier.predict(features_test)

#%%

#Print results 

print(confusion_matrix(label_test, label_predict))
print(classification_report(label_test, label_predict)) 

#%%
#Run new file 

test_data_features = test_data.iloc[:,1:6].values
new_classifier = classifier.predict(test_data_features)
test_data['new_labels'] = new_classifier

test_data.to_csv('test_data.csv')



