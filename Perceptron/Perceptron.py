from random import choice 
from numpy import array, dot, random, asmatrix, transpose
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import csv

unit = lambda x: -1 if x < 0 else 1

f = open("Winequality_dataset.csv", "r")
lines = f.readlines()

att1 = []
att2 = []
att3 = []
att4 = []
att5 = []
att6 = []
att7 = []
att8 = []
att9 = []
att10 = []
att11 = []
res = []
clas = []

for x in lines[1:]:
	
	att1.append(float(x.split(',')[0]))
	att2.append(float(x.split(',')[1]))
	att3.append(float(x.split(',')[2]))
	att4.append(float(x.split(',')[3]))
	att5.append(float(x.split(',')[4]))
	att6.append(float(x.split(',')[5]))
	att7.append(float(x.split(',')[6]))
	att8.append(float(x.split(',')[7]))
	att9.append(float(x.split(',')[8]))
	att10.append(float(x.split(',')[9]))
	att11.append(float(x.split(',')[10]))
	res.append(float(x.split(',')[11]))
	clas.append(float(x.split(',')[12]))

data = []

for x in range(len(att1)):
	data.append((array([att1[x], att2[x], att3[x], att4[x], att5[x], att6[x], att7[x], att8[x], att9[x], att10[x], att11[x]]), res[x], clas[x]))

train = []
test = []
	
for i in range(4500,6497):
	test.append(data[i])


print("Enter size of train data (shouldn't be more than 4500)- ")
n=int(raw_input())

for i in range(n):
	train.append(data[i])

mx = []
my = []

for i in range(len(train)):
	mx.append(train[i][0])

for i in range(len(train)):
	my.append(train[i][1])


y_train = asmatrix(my).astype("float")

w = transpose(pinv(mx) * transpose(y_train))


e_in = []
e_out = []
ete = 0.5

for i in range(300):
	
	errors = 0
	for j in range(len(train)): 
		x, dump, expected = train[j]
		if(expected == 0):
			expected = -1
		result = dot(w, x)	
		error = expected - unit(result)
		if(error!=0):
			errors+=1
	
	e_in.append(errors)
	
	errors = 0
	for j in range(len(test)): 
		x, dump, expected = test[j]
		if(expected == 0):
			expected = -1
		result = dot(w, x)
		error = expected - unit(result)
		if(error!=0):
			errors+=1
	
	e_out.append(errors)	
	
	for j in range(len(train)): 
		x, dump, expected = train[j]
		if(expected == 0):
			expected = -1
		result = dot(w, x)	
		error = expected - unit(result)
		w += error*ete*x

	
plt.xlim([0,300]) 
plt.title("Perceptron after Regression - Train count : 4500")
plt.xlabel("Number of iterations")
plt.ylabel("Error")

plt.plot(e_out, 'b', label="E_out")
plt.plot(e_in, 'r', label="E_in")
plt.legend(loc=1)
plt.show()
