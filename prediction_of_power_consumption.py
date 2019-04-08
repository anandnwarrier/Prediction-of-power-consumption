import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(5)


df4=pd.read_csv('sorted_redundancy_removed_zeros_removed_copy.csv',header=None)
x=np.array(df4,dtype=np.float64)
print(x)
l=len(x)-1

a=100
g=a
m_samples=l-a

Xd=np.zeros((m_samples,a))
y=np.zeros(m_samples)

print('l=',l,'m_samples=',m_samples)

for i in range(l):
	if i==m_samples:
		break;
	for j in range(i,i+a):
		Xd[i,j-i]=x[j][1]
	y[i]=x[i+a][1]


print('\n\nXd=',Xd,'\n\ny=',y)
df3=pd.DataFrame(Xd)
df3.to_csv('Xd.csv',float_format=np.float64)

b=m_samples

k=1-0.025
q=int(k*len(Xd))
x_train=Xd[:q]
y_train=y[:q]

x_test=Xd[q:]
y_test=y[q:]

print('ytrain',len(y_train),'\nytest',len(y_test))

#avgX=np.mean(Xd)

maxX=x[0][1]

for i in range(l):
    if x[i][1]>maxX :
        maxX=x[i][1]

maxX=maxX+10      ##################change: added 10 to max value for scaling down to the range (0,1)

X=(x_train)/maxX
#avgy=np.mean(y)

maxy=y[1]

for i in range(b):
    if y[i]>maxy :
        maxy=y[i]

maxy=maxy+10          ##################change: added 10 to max value for scaling down to the range (0,1)

Y=(y_train)/maxy  

t=40

model = Sequential()
model.add(Dense(50, input_dim=a, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, validation_split=1e-4, epochs=t, batch_size=10, verbose=1)

train=model.predict(X)
train=train*maxy
y_train=y_train


plt.plot(train[:150],color='b',label='predicted')
plt.plot(y_train[:150],color='r',label='ground truth')
plt.xlabel('time index')
plt.ylabel('Power (in kWh)')
plt.title('Training results FOR FIRST 150 samples:\n(plotting all will make the plot crowded)')
plt.legend()
plt.show()

X2=x_test/maxX
predictions = model.predict(X2)
predictions=predictions*maxy

plt.plot(predictions,color='b',label='predicted')
plt.plot(y_test,color='r',label='ground truth')
plt.xlabel('time index')
plt.ylabel('Power (in kWh)')
plt.legend()
plt.title('Testing results')
plt.show()
print(y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

