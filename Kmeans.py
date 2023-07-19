from sklearn.cluster import KMeans
import csv
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import scipy
# %matplotlib inline

#id	Age	Sex	ChestPain	RestBP	Chol	Fbs	RestECG	MaxHR	ExAng	Oldpeak	Slope	Ca	Thal	AHD

data = pd.read_csv('heart.csv')
print(data.head(303))

names=list(data.columns)

#age
youngAge_index=data[(data.age<55)].index
elderlyAge_index=data[(data.age>=55)].index
for index in elderlyAge_index:
    data.loc[index,'age']=1
    
for index in youngAge_index:
    data.loc[index,'age']=0

#thalach
lowthalach_index=data[(data.thalach<60)].index
medthalach_index=data[(data.thalach>=60)&(data.thalach<=100)].index
hithalach_index=data[(data.thalach>100)].index

for index in hithalach_index:
    data.loc[index,'thalach']=2
    
for index in medthalach_index:
    data.loc[index,'thalach']=1

for index in lowthalach_index:
    data.loc[index,'thalach']=0

#trestbps
highTrestbps_index=data[(data.trestbps>130)].index
idealTrestbps_index=data[(data.trestbps>=90)].index & data[(data.trestbps<=130)].index
lowTrestbps_index=data[(data.trestbps<90)].index
for index in highTrestbps_index:
    data.loc[index,'trestbps']=2
    
for index in idealTrestbps_index:
    data.loc[index,'trestbps']=1

for index in lowTrestbps_index:
    data.loc[index,'trestbps']=0

#chol
lowChol_index=data[(data.chol<200)].index
medChol_index=data[(data.chol>=200)&(data.chol<=240)].index
hiChol_index=data[(data.chol>240)].index

for index in hiChol_index:
    data.loc[index,'chol']=2
    
for index in medChol_index:
    data.loc[index,'chol']=1

for index in lowChol_index:
    data.loc[index,'chol']=0

#oldpeak
lowOP_index=data[(data.oldpeak<1.5)].index
medOP_index=data[(data.oldpeak>=1.5)&(data.oldpeak<=2.55)].index
hiOP_index=data[(data.oldpeak>2.55)].index

for index in hiOP_index:
    data.loc[index,'oldpeak']=2
    
for index in medOP_index:
    data.loc[index,'oldpeak']=1

for index in lowOP_index:
    data.loc[index,'oldpeak']=0
print(data.shape)	

data.to_csv('newheart.csv')

print(data.head(303))
y = data.target.values

drop = ['target']
data = data.drop(drop, axis = 1)

# print(data.shape)
# correlations = data.corr()
# # plot correlation matrix
# fig = plt.figure()
# fig.canvas.set_window_title('Correlation Matrix')
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,9,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
#fig.savefig('Correlation Matrix.png')
    
 
# #scatterplot
# scatter_matrix(data)
    
# plt.show()

# ncols=3
# plt.clf()
# f = plt.figure(1)
# f.suptitle(" Data Histograms", fontsize=12)
# vlist = list(data.columns)
# nrows = len(vlist) // ncols
# if len(vlist) % ncols > 0:
# 	nrows += 1
# for i, var in enumerate(vlist):
# 	plt.subplot(nrows, ncols, i+1)
# 	plt.hist(data[var].values, bins=15)
# 	plt.title(var, fontsize=10)
# 	plt.tick_params(labelbottom='off', labelleft='off')
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# plt.show()

# fulldata = pd.get_dummies(fulldata,columns=['target'])

k=3
kmeans = KMeans(n_clusters=k).fit(data)
labels = kmeans.labels_
print(labels)
centroids = kmeans.cluster_centers_
print("Centroids Value")
print(centroids)    
data['oldtarget'] = y	
data['categories'] = kmeans.labels_

ay1=[]
ay2=[]
ay3=[]

N=1
for i in range(0,len(centroids[0])):
    ay2 = np.append(ay2,0.6+0.6*np.random.rand(N))
    ay1 = np.append(ay1,0.4+0.3*np.random.rand(N))
    ay3 = np.append(ay3,0.3*np.random.rand(N))

datax = (centroids[0],centroids[1],centroids[2])
datay = (ay1,ay2,ay3)
colors = ("red", "green", "blue","yellow")
groups = ("Cluster1", "Cluster2", "Cluster3") 
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
 
for datx, daty, color, group in zip(datax,datay, colors, groups):
    x, y = datx,daty
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
 
plt.title('Cluster Centroids')
plt.legend(loc=2)
plt.show()


clusters = {}
n = 0

#transpose matrices
# y = y.T

# def find_permutation(n_clusters, real_labels, labels):
#     permutation=[]
#     for i in range(n_clusters):
#         idx = labels == i
#         new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
#         permutation.append(new_label)
#     return permutation

# permutation = find_permutation(3,y,labels)
# print(permutation) 

for item in labels:
	if item in clusters:
		clusters[item].append(data.iloc[n])
	else:
		clusters[item] = [data.iloc[n]]
	n +=1
high=0
low=0
for i in range(len(clusters[0])):
    if clusters[0]['oldtarget'] == 1:
        high+=1
    else:
        low+=1
print(high)
print(low)
with open('Cluster0.csv','w',newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for line in clusters[0]:
    	writer.writerow(line)
    	
with open('Cluster1.csv','w',newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for line in clusters[1]:
    	writer.writerow(line)	

with open('Cluster2.csv','w',newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for line in clusters[2]:
    	writer.writerow(line)	
   
print("Finished")    	   	