from random import choice 
from numpy import array, dot, random, asmatrix, transpose
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import csv

f = open("winequality-white.csv", "r")
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

for x in lines[1:]:
	
	att1.append(float(x.split(';')[0]))
	att2.append(float(x.split(';')[1]))
	att3.append(float(x.split(';')[2]))
	att4.append(float(x.split(';')[3]))
	att5.append(float(x.split(';')[4]))
	att6.append(float(x.split(';')[5]))
	att7.append(float(x.split(';')[6]))
	att8.append(float(x.split(';')[7]))
	att9.append(float(x.split(';')[8]))
	att10.append(float(x.split(';')[9]))
	att11.append(float(x.split(';')[10]))
	res.append(float(x.split(';')[11]))

data = []

for x in range(len(att1)):
	data.append((array([att1[x], att2[x], att3[x], att4[x], att5[x], att6[x], att7[x], att8[x], att9[x], att10[x], att11[x]]), res[x]))

num = [30,40,50,60,70]
err = []

train = []
test = []

n=(30*4898)/100

for i in range(n):
	train.append(data[i])
	
for i in range(n,4898):
	test.append(data[i])


mx = []
my = []

for i in range(len(train)):
	mx.append(train[i][0])

for i in range(len(train)):
	my.append(train[i][1])


y_train = asmatrix(my).astype("float")

w = transpose(pinv(mx) * transpose(y_train))

sum = 0

for i in range(len(test)):
	x, expected = test[i]
	result = dot(w, x)	
	sum += (result-expected)*(result-expected)

e1 = float(sum/3430)

print(e1)

err.append(e1)

train = []
test = []

n=(40*4898)/100


for i in range(n):
	train.append(data[i])
	
for i in range(n,4898):
	test.append(data[i])


mx = []
my = []

for i in range(len(train)):
	mx.append(train[i][0])

for i in range(len(train)):
	my.append(train[i][1])


y_train = asmatrix(my).astype("float")

w = transpose(pinv(mx) * transpose(y_train))

sum = 0

for i in range(len(test)):
	x, expected = test[i]
	result = dot(w, x)	
	sum += (result-expected)*(result-expected)

e2 = float(sum/3430)

print(e2)

err.append(e2)

train = []
test = []

n=(50*4898)/100

for i in range(n):
	train.append(data[i])
	
for i in range(n,4898):
	test.append(data[i])


mx = []
my = []

for i in range(len(train)):
	mx.append(train[i][0])

for i in range(len(train)):
	my.append(train[i][1])


y_train = asmatrix(my).astype("float")

w = transpose(pinv(mx) * transpose(y_train))

sum = 0

for i in range(len(test)):
	x, expected = test[i]
	result = dot(w, x)	
	sum += (result-expected)*(result-expected)

e3 = float(sum/3430)

print(e3)

err.append(e3)

train = []
test = []

n=(60*4898)/100

for i in range(n):
	train.append(data[i])
	
for i in range(n,4898):
	test.append(data[i])


mx = []
my = []

for i in range(len(train)):
	mx.append(train[i][0])

for i in range(len(train)):
	my.append(train[i][1])


y_train = asmatrix(my).astype("float")

w = transpose(pinv(mx) * transpose(y_train))

sum = 0

for i in range(len(test)):
	x, expected = test[i]
	result = dot(w, x)	
	sum += (result-expected)*(result-expected)

e4 = float(sum/3430)

print(e4)

err.append(e4)

train = []
test = []

n=(70*4898)/100


for i in range(n):
	train.append(data[i])
	
for i in range(n,4898):
	test.append(data[i])


mx = []
my = []

for i in range(len(train)):
	mx.append(train[i][0])

for i in range(len(train)):
	my.append(train[i][1])


y_train = asmatrix(my).astype("float")

w = transpose(pinv(mx) * transpose(y_train))

sum = 0

for i in range(len(test)):
	x, expected = test[i]
	result = dot(w, x)	
	sum += (result-expected)*(result-expected)

e5 = float(sum/3430)

print(e5)

err.append(e5)

print(num)

plt.xticks(num) 
plt.title("Linear Regression")
plt.xlabel("Train data percent")
plt.ylabel("Error")

plt.plot(num, err, 'b')
plt.show()
