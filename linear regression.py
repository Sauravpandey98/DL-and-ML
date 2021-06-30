import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 


data= np.loadtxt('ex1data2.txt', delimiter=',', unpack=True)# data here


features=len(data)-1 #printing no. of features(len(numpyarray) print no. of rows)
Standardisation = preprocessing.StandardScaler()
x=np.transpose(np.matrix(data[0:features,:]))
scalar=Standardisation.fit(x)
scaled_x=scalar.transform(x) 
Y=np.transpose(np.matrix(np.array(data[features,:])))
m=len(x)
alpha=0.001       #set the values of learning rate
iteration=150     #no. of iteration
X=np.c_[np.ones(m),scaled_x]

theta=np.matrix(np.zeros(features+1))
theta1=np.transpose(theta)


#function for computing cost
def compute_cost(x,y,theta):
    j=(1/(2*len(x)))*(np.matmul(np.transpose(np.matmul(X,theta)-Y),(np.matmul(X,theta)-Y)))
    return j


# Gradient Descent
def gradient_desc(x,y,theta,alpha,iteration):
	J=np.zeros(iteration)
	for i in range(0,iteration):
		theta=theta-(alpha*((1/len(x))*(np.matmul((np.transpose(X)),X@theta-Y))))
		J[i]=compute_cost(x,y,theta)
	return theta

# Observing Cost	   
def observe_cost():#function for observing the convergence of cost function with a given learning rate and iteration
    fig= plt.figure(figsize=(10,10))
    axes= fig.add_axes([1,1,1,1])

    axes.plot(gradient_desc(X,Y,theta1,alpha,iteration))
 #   plt.xticks(range(0,iteration))
 #   plt.yticks(np.arange(0,8,0.5))
    plt.show()

#for checking relation between given values and predicted values  
def cross_relation():	
	fig= plt.figure(figsize=(10,10))
	axes=fig.add_axes([1,1,1,1])   

	axes.scatter(data[1,:],data[2,:],marker="x")
	predicted_y=X@gradient_desc(X,Y,theta1,alpha,iteration)
	print(predicted_y)
	axes.plot(data[1,:],predicted_y,'-r')
	plt.show()
    
#for predicting y   
def predict(y):
	y=(np.array(y)).reshape(1,-1)
	y_scaled=scalar.transform(y)
	predicted=(np.c_[np.ones(1),y_scaled])@gradient_desc(X,Y,theta1,alpha,iteration)
	return predicted

