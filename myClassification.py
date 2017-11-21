import numpy as np 
from sklearn.neural_network import MLPClassifier

# calculate line layer from point (x1,x2)
def lineLayer(x1,x2,coef):
	v = np.zeros((len(coef)))
	for i in range(len(coef)) :
		l = coef[i]
		v[i]=l[2]*x2 - l[0]*x1 - l[1]
	return v


# activation function (step)
def phi(v):
	y = np.zeros(v.shape)
	for i in range(len(v)):
		if (v[i] >= 0):
			y[i] = 1
		else :
			y[i] =0
	return y


# calculate lines equations	
coef = np.ones((9,3))

# v1
p1 = [ -3. , 0]
p2 = [ 2.5 , 4]
coef[0,:2] = np.polyfit(p1, p2, 1)



# v2 
p1 = [ -3. , -3]
p2 = [ 2.5 , 0 ]
coef[1] = [-1,-3,0]

#v3
p1 = [ 1. , 3 ]
p2 = [ 0,0 ]
coef[2] = [0,0,1]

#v4
p1 = [ 0 , 0 ]
p2 = [ 1 , 2 ]
coef[3] = [-1 ,0,0]

#v5 
p1 = [ 0. , 2. ]
p2 = [ 2. , 2. ]
coef[4,:2] = np.polyfit(p1, p2, 1)

#v6
p1 = [ 2. , 2. ]
p2 = [ 0 , 2 ]
coef[5] = [-1 ,2,0]

#v7
p1 = [ 3 , 3]
p2 = [ 2.5 , 4 ]
coef[6] = [-1,3,0]

#v8
p1 = [ 3. , 5]
p2 = [ 2.5 , 2.5 ]
coef[7,:2] = np.polyfit(p1, p2, 1)

# v9
p1 = [ 3 , 5 ]
p2 = [ 4 , 2.5 ]
coef[8,:2] = np.polyfit(p1, p2, 1)




# binary permutaions 
def per(n):
	X = np.zeros((2**n,n))
	for i in range(1<<n):
		s=bin(i)[2:]
        	s='0'*(n-len(s))+s
        	X[i] = map(int,list(s))
        
	return X

X= per(9)

# init training labels
Y = np.ones((2**9,2))

# entering the classes into the Y labels
for i in range(len(X)):
	pos = X[i]
	if(not(pos[0]) and pos[1] and pos[2] and not(pos[3]) ):
		Y[i] = [1,0]
	elif (pos[2] and pos[3] and not(pos[4]) and not(pos[5]) ):
		Y[i] = [0,1]
	elif(pos[6] and pos[7] and not(pos[8])):
		Y[i] = [0,0]


# neural net 
clf = MLPClassifier()
clf.fit(X,Y)


# 
while True :
	x1 = input("Enter x1 ")
	x2 = input("Enter x2 ")
	v = lineLayer(x1,x2,coef)
	y = np.matrix(phi(v))
	print "the class is " clf.predict(y)

