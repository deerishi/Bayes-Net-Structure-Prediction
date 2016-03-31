import numpy as np
from copy import copy

data=np.load('ChessData.npy')

#0 and 1 indexes have t and f as boolean discrete values values

#now we have to find the condtional probability

Z=['won','nowin']
X=['t','f']
Y=['b', 'w', 'n']

def findConditionalProbability2Variables(indexX,valueX,indexZ,valueZ):
    '''
    Here we find the condtional probabilities P(X|Z) and P(Y|Z)
    P(X|Z)=#count(X&Z)/#count(Z)
    '''
    rows=np.where(data[:,indexZ]==valueZ)[0]
    arr=copy(data[rows])
    denominator=arr.shape[0]
    rows=np.where(arr[:,indexX]==valueX)[0]
    arr2=copy(arr[rows])
    
    numerator=arr2.shape[0]
    return numerator,denominator,float(numerator+1)/(denominator+2)
    

    
def findProbability3Variables(indexX,valueX,indexY,valueY,indexZ,valueZ):
        #we find P(X,Y,Z)
        #The maximum lkelyhood estimate is #(x&y&Z)/totalnumber of datapoints
        #print 'val'
    rows=np.where(data[:,indexZ]==valueZ)[0] 
    arr=copy(data[rows])
        #print 'arr is ',arr.shape
        #now we have the array for all thise values
    rows=np.where(arr[:,indexY]==valueY)[0]
    #print 'rows is ',rows
    arr=copy(arr[rows])
    #now taking all training vectors with X     
    rows=copy(np.where(arr[:,indexX]==valueX)[0])
    #print 'rows is ',rows,' len is ',len(rows)
    
    count=0
    if len(rows)>0:
        arr=copy(arr[rows])
        count=arr.shape[0]
    else:
        count=0;
        #now we have that array 
    #print '\nFor X= ',x,' Y= ',y,' Z= ',z,' Numerator is ',count,' denominator is ',data.shape[0]   
    return  float(count+1)/(data.shape[0]+2) # adding a Beta Prior
    

def findConditionalProbability3Variables(indexX,valueX,indexY,valueY,indexZ,valueZ):
    
    prob=findProbability3Variables(indexX,valueX,indexY,valueY,indexZ,valueZ)
    
    rows=np.where(data[:,indexZ]==valueZ)
    arr=copy(data[rows])
    
    prob2=float(arr.shape[0])/data.shape[0]
    
    return (prob,prob/prob2)
    

mi=0    
i1=10
i2=14
for z in Z:
    for x in X:
        numX,denX,probXCon=findConditionalProbability2Variables(i1,x,data.shape[1]-1,z)
        for y in Y:
            #prob3Variables=findProbability3Variables(0,x,1,y,data.shape[1]-1,z)
            prob3Variables,prob3VariablesCon=findConditionalProbability3Variables(i1,x,i2,y,data.shape[1]-1,z)
            numY,denY,probYCon=findConditionalProbability2Variables(i2,y,data.shape[1]-1,z)
            
            print '\nFor X= ',x,' Y= ',y,' Z= ',z,' we have prob3Variables is ',prob3Variables,' prob3VariablesCon ',prob3VariablesCon,' probXCon is ',probXCon,' probYCon ',probYCon,'\n'
            mi+=prob3Variables*(np.log2(prob3VariablesCon/(probXCon*(probYCon))))
            

print '\n\n mi is ',mi
            
            
            
            
            
