import numpy as np

'''
basically we need to change the array indice 14 to a set of 3 other binary variables so that we can treat the entire training dataset as a 
Bernoulli document model. This is being done to handle categorical features and to avoid mixing up Bernoilli Document model and Multinomial Model
'''
d={'t':1,'f':0,'l':1,'g':0,'1':1,'0':0,'n':0}

def read(file1):
    l1=[]
    f2=open('ChessProcessed.txt','w')
    f=open(file1,'r')
    j=1
    #1)First create the entire list of feature vectors then write it to the file1
    for line in f:
        print 'Processing j=',j
        j+=1
        line=line.rstrip()
        chars=line.split(',')
        #now chars is a list of all the feature vectors
        for i in range(0,len(chars)):
            if i==14:
                if chars[i]=='b':
                    f2.write('1,0,0,')
                elif chars[i]=='w':
                    f2.write('0,1,0,')
                else:
                    f2.write('0,0,1,')
            else:
                if i!=len(chars)-1:
                    f2.write(str(d[chars[i]])+',')
                else:
                    f2.write(str(d[chars[i]])+'\n')
        
print 'Complete'    
read('Possible Datasets/Chess (King-Rook vs. King-Pawn) Data Set.data')
