import numpy as np
from copy import copy

def read(file):
    f=open(file,'r')
    l=[]
    j=1
    
    for line in f:
        line=line.rstrip()
        chars=line.split(',')
        print 'Processing j=',j
        j+=1
        l.append(chars)
        
    arr=np.asarray(l)
    return arr
    
    
arr=read('Chess End Game Data/ChessProcessed.txt')
print 'arr is ',arr
f1='Chess End Game Data/ChessFull'
f2='Chess End Game Data/ChessArray'
f3='Chess End Game Data/ChessLabels'

np.save(f1,arr)

arr2=copy(arr[:,0:arr.shape[1]-1])
np.save(f2,arr2)
labels=copy(arr[:,arr.shape[1]-1])
np.save(f3,labels)
print 'shape is ',arr.shape
print 'labels are ',labels.shape
print 'arr is ',arr2.shape

for i in range(0,arr.shape[1]):
    s1=set(arr[:,i])
    print 'i= ',i,' s1 contains ',s1,' of length ',len(s1)
    

