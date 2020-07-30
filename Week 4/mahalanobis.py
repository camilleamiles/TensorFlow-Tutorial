import pandas as pd
import scipy as sp
import numpy as np
import tensorflow as tf 
import eagerpy as ep
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import LinfDeepFoolAttack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



means=pd.read_csv('mnist_mean_actv1000.csv')
meanarraylist=[]
meanarraylist.append(np.array(means['zero_mean_activation']))
meanarraylist.append(np.array(means['one_mean_activation']))
meanarraylist.append(np.array(means['two_mean_activation']))
meanarraylist.append(np.array(means['three_mean_activation']))
meanarraylist.append(np.array(means['four_mean_activation']))
meanarraylist.append(np.array(means['five_mean_activation']))
meanarraylist.append(np.array(means['six_mean_activation']))
meanarraylist.append(np.array(means['seven_mean_activation']))
meanarraylist.append(np.array(means['eight_mean_activation']))
meanarraylist.append(np.array(means['nine_mean_activation']))

data2=pd.read_csv('mnist_stdev_actv1000.csv')
stddevlist=[]
stddevlist.append(np.array(data2['zero_stdev_activation']))
stddevlist.append(np.array(data2['one_stdev_activation']))
stddevlist.append(np.array(data2['two_stdev_activation']))
stddevlist.append(np.array(data2['three_stdev_activation']))
stddevlist.append(np.array(data2['four_stdev_activation']))
stddevlist.append(np.array(data2['five_stdev_activation']))
stddevlist.append(np.array(data2['six_stdev_activation']))
stddevlist.append(np.array(data2['seven_stdev_activation']))
stddevlist.append(np.array(data2['eight_stdev_activation']))
stddevlist.append(np.array(data2['nine_stdev_activation']))

from keract import get_activations, display_activations
keract_inputs=input_test[:1]
keract_target=target_test[:1]


#group mnist examples
#target_train[:5]
trainlabels=np.argmax(target_train, axis=1)

zero_indices=np.where(trainlabels==0)[0]
one_indices=np.where(trainlabels==1)[0]
two_indices=np.where(trainlabels==2)[0]
three_indices=np.where(trainlabels==3)[0]
four_indices=np.where(trainlabels==4)[0]
five_indices=np.where(trainlabels==5)[0]
six_indices=np.where(trainlabels==6)[0]
seven_indices=np.where(trainlabels==7)[0]
eight_indices=np.where(trainlabels==8)[0]
nine_indices=np.where(trainlabels==9)[0]

zero_indices=zero_indices[0:100]
one_indices=one_indices[0:100]
two_indices=two_indices[0:100]
three_indices=three_indices[0:100]
four_indices=four_indices[0:100]
five_indices=five_indices[0:100]
six_indices=six_indices[0:100]
seven_indices=seven_indices[0:100]
eight_indices=eight_indices[0:100]
nine_indices=nine_indices[0:100]

indices=[zero_indices,one_indices,two_indices,three_indices,four_indices,five_indices,six_indices,seven_indices,eight_indices,nine_indices]


images=[]
labels=[]
for indexarr in indices:
	for anindex in indexarr:
		images.append(input_train[anindex])

for anum in range(10):
	for i in range(100):
		labels.append(anum)

fmodel=TensorFlowModel(model, bounds=(0,1))
attack=LinfDeepFoolAttack()
#epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
#_, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.1)
_, advsDF, success = attack(fmodel, tf.convert_to_tensor(images), tf.convert_to_tensor(labels), epsilons=0.03)


advsinputs=np.array(advsDF) # 100, 28, 28, 1

#first trial of PCA...just looked at zero class
x=[]
for anindex in range(len(images[0:100])):
	#for i in range(len(10)):
	keract_inputs=images[anindex]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
	valarray=activations['dense128'][0]
	x.append(valarray)


#just looking at zero class
x0=np.array(x)
pca=PCA(n_components=30)
z=pca.fit(x0)
howmuchvar=np.sum(z.explained_variance_ratio_)
z0=pca.transform(x0)


#PCA for each class
b=0
a=0
x=[]
actdict={}
for anindex in range(len(images)):
	#for i in range(len(10)):
	keract_inputs=images[anindex]
	activations=get_activations(model, keract_inputs.reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
	valarray=activations['dense128'][0]
	x.append(valarray)

	b+=1
	if b%100==0:
		activations=get_activations(model, advsinputs[0].reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
		valarray=activations['dense128'][0]
		x.append(valarray)
		actdict[a]=x #all 100 zero example activations
		x=[]
		a+=1
	else:
		continue

#woohoo! actdict stores all activations for 100 examples of each class..



#could have done loop but just did manually

x0=np.array(actdict[0]) #x0.shape (100,128)
pca=PCA(n_components=40)
z0=pca.fit_transform(x0)   #do this for all
howmuchvar=np.sum(z.explained_variance_ratio_) 
z0=pca.transform(x0) #z0.shape (100,35)

x1=np.array(actdict[1]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x1)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z1=pca.transform(x1) #shape (100,35)

x2=np.array(actdict[2]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x2)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z2=pca.transform(x2)

x3=np.array(actdict[3]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x3)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z3=pca.transform(x3)

x4=np.array(actdict[4]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x4)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z4=pca.transform(x4)

x5=np.array(actdict[5]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x5)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z5=pca.transform(x5)

x6=np.array(actdict[6]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x6)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z6=pca.transform(x6)

x7=np.array(actdict[7]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x7)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z7=pca.transform(x7)

x8=np.array(actdict[8]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x8)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z8=pca.transform(x8)

x9=np.array(actdict[9]) #x0.shape (100,128)
pca=PCA(n_components=40)
z=pca.fit(x9)
howmuchvar=np.sum(z.explained_variance_ratio_) 
z9=pca.transform(x9)

#zero means
z0_mean=np.zeros((40))
for i in range(0, len(z0)):
	z0_mean=z0_mean+z0[i]
z0_mean=z0_mean/100

#zero cov
cov0=np.cov(z0.T)
invcov0=sp.linalg.inv(cov0)








#y=meanarraylist[0]

#mahal stuff
#x= activations(advsinputs[0])
activations=get_activations(model, advsinputs[0].reshape(1,28,28,1), layer_names='dense128', output_format='simple')		
valarray=activations['dense128'][0]
x=valarray #needs 128 images with 128 neurons each
y=meanarraylist[0]
#x_minus_mu=x-meanarraylist[0]

cov=np.cov(x)
inv_covmat=sp.linalg.inv(cov)
left_term=np.dot(x_minus_mu, inv_covmat)
mahal=np.dot(left_term, x_minus_mu.T)
mahal.diagonal()


x=valarray
y=meanarraylist[0]
cov_mat=np.stack((x,y),axis=1)
np.cov(cov_mat)
