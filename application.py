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