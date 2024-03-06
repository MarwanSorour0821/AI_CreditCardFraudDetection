from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns 
from scipy.stats import iqr
from sklearn.tree import DecisionTreeClassifier

#Data set that I will be wokring on
df = pd.read_csv("/Users/marwansorour/Desktop/AI_CreditCardFraudDetection/creditcard.csv")
print(df.head())

#plot a counterplot to see the balance of the data set
fig, ax = plt. subplots(figsize=(6, 4))
ax = sns.countplot(x= 'Class', data=df)
plt.tight_layout()
plt.show()

#Plotting to visualise the count VS the time for all data

#convert time from seconds to hours ot visualise the plot
time_in_hours = df['Time']/3600

plt.figure(figsize=(12,6), dpi = 80)
sns.distplot(time_in_hours, bins=48, kde=False)
plt.xticks(np.arange(0,54,6))
plt.xlabel('Time After First Transaction (hr)')
plt.ylabel('Count')
plt.title('Transaction Times')
plt.show()


#Filtering out the real transactions from the fraud transactions
df_real_transactions = df[df['Class'] == 0]
non_fraud_time_in_hours = df_real_transactions['Time']/3600
plt.figure(figsize=(12,6))
sns.distplot(non_fraud_time_in_hours, bins=48)
plt.xticks(np.arange(0,54,6))
plt.xlabel('Time After First Transaction (hr)')
plt.title('Non-Fraud Transactions')
plt.ylabel('Count')
plt.show()

fraud_time_in_hours = df['Class']
#Filter fraud transactions
df_fraud_transactions = df[df['Class'] == 1]
fraud_time_in_hours = df_fraud_transactions['Time']/3600
plt.figure(figsize=(12,6))
sns.distplot(fraud_time_in_hours, bins=48)
plt.xticks(np.arange(0,54,6))
plt.xlabel('Time After First Transactions (hr)')
plt.ylabel('Count')
plt.title('Fraud Transactions')
plt.show()

#Plot the Amount
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(20,6))
sns.distplot(df['Amount'], ax=ax1)
sns.boxplot(x=df['Amount'], ax=ax2)
plt.show()

#Perform IQR analysis to remove outliers
upper_limit = df['Amount'].quantile(0.75) + (1.5*iqr(df['Amount']))
print(upper_limit)
print(df[df.Amount > upper_limit]['Class'].value_counts())

#Removing Outliers
df_copy = df[df.Amount <= 8000]
print(df_copy['Class'].value_counts())
print('\nPercentage of Fraudulent Activity: {:.2%}'.format((df_copy[df_copy['Class']==1].shape[0]/df_copy.shape[0])))

fig1, (ax3,ax4) = plt.subplots(2,1, figsize=(10,15))
sns.boxplot(x=df_copy[df_copy['Class'] == 1].Amount, ax=ax3)
ax3.set_title('Fraudulent transactions')
sns.boxplot(x=df_copy[df_copy['Class'] == 0].Amount, ax=ax4)
ax4.set_title('Real Transactions')
plt.show()


#=====================Understand Correlation Between the Features in the Data
corr= df.corr()
fig,ax = plt.subplots(figsize=(9,7))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.1,cmap='RdBu', ax=ax)
plt.tight_layout()
plt.show()

#Process to balance the data since it is higly imbalanced
#We will use undersampling
non_fraud = df_copy[df_copy['Class']==0].sample(2000)
fraud = df_copy[df_copy['Class']==1]
print(len(non_fraud), len(fraud))

df_2 = pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)
x = df_2.drop(['Class'], axis=1).values
y=df_2["Class"].values

#Create training data set
p = TSNE(n_components=2, random_state=24).fit_transform(x)

color_map = {0:'red', 1:'blue'}
plt.figure()
for index, cl in enumerate(np.unique(y)):
    plt.scatter(x=p[y==cl,0], y = p[y==cl,1], c=color_map[index], label=cl)

#Data for fraid and on-fraud data overlap which is difficutl to comrpehend
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualisation of test data')
plt.show()

#To solve the issue we will use an autoencoder for fraud detection
x_scale = preprocessing.MinMaxScaler().fit_transform(x)
x_norm, x_fraud = x_scale[y==0], x_scale[y==1]

nan_indices = np.isnan(x_norm).any(axis=1)

# Remove rows with NaN values from x_norm
x_norm_filtered = x_norm[~nan_indices]

#Architecutre of autoencoder: 6 layers
#input layer
#4 hidden layers
#Output layer

#Constructing the autoencoder
#tanh and relu are layers
autoencoder = Sequential()
autoencoder.add(Dense(x.shape[1], activation='tanh'))
autoencoder.add(Dense(100, activation='tanh'))
autoencoder.add(Dense(50, activation='relu'))
autoencoder.add(Dense(50,activation='tanh'))
autoencoder.add(Dense(100, activation='tanh'))
autoencoder.add(Dense(x.shape[1], activation='relu'))

autoencoder.compile(optimizer='adadelta', loss="mse")
autoencoder.fit(x_norm, x_norm, batch_size=256, epochs=10, shuffle=True, validation_split=0.2)

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])

norm_hid_rep = hidden_representation.predict(x_norm)
fraud_hid_representation = hidden_representation.predict(x_fraud)

rep_x = np.append(norm_hid_rep, fraud_hid_representation, axis=0)
y_n = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(fraud_hid_representation.shape[0])
rep_y = np.append(y_n, y_f)


#Now the data is more comprehendable as the data is now not overlapping
#using dimensioanlity reduction
p2 = TSNE(n_components=2, random_state=24).fit_transform(rep_x)
color_map = {0:'red', 1:'blue'}
plt.figure()
for index, cl in enumerate(np.unique(y)):
    plt.scatter(x=p2[y==cl,0], y = p2[y==cl,1], c=color_map[index], label=cl)

#Data for fraud and on-fraud data overlap which is difficutl to comrpehend
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualisation of test data')
plt.show()

#TRAIN THE MODELS ON THE EXTRACTED DATA
x_train, val_x, y_train, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
clf = LogisticRegression(solver="lbfgs").fit(x_train, y_train)
predict_y = clf.predict(val_x)

#Get classification report
print(classification_report(val_y, predict_y))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(val_y, predict_y, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")
plt.show()

conf_matrix = confusion_matrix(val_y, predict_y)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print("Accuracy:", accuracy)


model_tree = DecisionTreeClassifier(max_depth=4, criterion="entropy")
model_tree.fit(x_train, y_train)
y_pred_tree = model_tree.predict(val_x)

print(classification_report(val_y, y_pred_tree))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(val_y, y_pred_tree, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix") 
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")
plt.show()

conf_matrix = confusion_matrix(val_y, y_pred_tree)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print("Accuracy:", accuracy)
