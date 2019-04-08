import numpy as np
import pandas as pd

np.random.seed(4)

"""

0. Timestamp (epoch time)
1. Text file name, sample set
2. Phase A voltage
3. Phase B voltage
4. Phase C voltage
5. Phase A current
6. Phase B current
7. Phase C current
8. Phase A active power
9. Phase B active power
10.Phase C active power
11. Average Voltage
12. Average Curent
13. Total Active power
14. Line frequency
15. Cumulative KWH energy
16. Power in mega/ kilo (00 kilo and 01 mega)
17. Total power factor

epoch,sample_set,Va,Vb,Vc,Ia,Ib,Ic,Pa,Pb,Pc,Vavg,Iavg,Pact,f ,Ecum_kwh,mega,Pf
  0  ,    1     ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, 11 , 12 , 13 ,14,   15   , 16 ,17

"""

df=pd.read_csv('total.csv',header=None)
print(df)

epoch_df=df[0]
pow_df=df[13]


df1=df[[0,13]]
X=np.array(df1,dtype=np.float64)
print(X)
N=len(X)


for i in range(N-1):
    for j in range(N-i-1):
        if(X[j][0]>X[j+1][0]):
            temp1=X[j+1][0]
            temp2=X[j+1][1]
            X[j+1][0]=X[j][0]
            X[j+1][1]=X[j][1]
            X[j][0]=temp1
            X[j][1]=temp2

df2=pd.DataFrame(X)
df2.to_csv('sorted.csv',float_format=np.float64)

print(X)

x=np.zeros((N,2))

x[0][0]=X[0][0]
x[0][1]=X[0][1]

j=1
for i in range(N):
    if((X[i][0]!=x[j-1][0])&(X[i][1]!=0)): #changed from: if(X[i][0]!=x[j-1][0])
        x[j][0]=X[i][0]
        x[j][1]=X[i][1]
        j+=1

l=j

print(x,l)
df3=pd.DataFrame(x)
df3.to_csv('sorted_redundancy_removed_zeros_removed.csv',float_format=np.float64)
