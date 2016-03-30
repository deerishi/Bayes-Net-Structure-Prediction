import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot


#we need to usea beta prior of ~Beta(2,2) since Z is a bernoulli rndom variable , so we need to add the halllucinated counts 
#simply add +2 to the count of P(Z) in the denominator and ad +1 in the numberator for eveything 
np.set_printoptions(threshold='nan')
class TAN:
    def __init__(self,file1):  
        self.variableValues={}
        self.X_train=self.loadData(file1)
        self.labels=set(self.X_train[:,-1])
        for i in range(0,self.X_train.shape[1]-1):
            s1=set(self.X_train[:,i])
            self.variableValues[i]=s1
        
        #print 'X_train is ',self.X_train
        print 'the possible variables values are ',self.variableValues
        #print 'labels are ',self.labels
        self.numAttributes=self.X_train.shape[1]-1 #we innore the labels
        print 'now saving'
        np.save('ChessData',self.X_train)
        self.calculateConditionalMutualInformation()
        
        self.createMaximumSpanningTree()
        print 'the GRAPH mst is ',self.graphMst,'\n'
        self.plotGraph()
    
    
    def findConditionalProbability3Variables(self,indexX,valueX,indexY,valueY,indexZ,valueZ):
        
        #Here we find P(X,Y|Z)= #count(X&Y&Z)/#count(Z)
        
        countNumerator,temp=self.findProbability3Variables(indexX,valueX,indexY,valueY,indexZ,valueZ)
        
        arr=np.where(self.X_train[:,indexZ]==valueZ)[0]
        
        return float(countNumerator)/(arr.shape[0]+2) # accounting for beta distibution
        
    def findConditionalProbability2Variables(self,indexX,valueX,indexZ,valueZ):
        '''
        Here we find the condtional probabilities P(X|Z) and P(Y|Z)
        P(X|Z)=#count(X&Z)/#count(Z)
        '''
        
        rows=np.where(self.X_train[:,indexZ]==valueZ)[0]
        arr=copy(self.X_train[rows])
        denominator=arr.shape[0]
        rows=np.where(arr[:,indexX]==valueX)[0]
        arr=copy(arr[rows])
        return float(arr.shape[0]+1)/(denominator+2)
        
        
        
    def findProbability3Variables(self,indexX,valueX,indexY,valueY,indexZ,valueZ):
        #we find P(X,Y,Z)
        #The maximum lkelyhood estimate is #(x&y&Z)/totalnumber of datapoints
        #print 'val'
        rows=np.where(self.X_train[:,indexZ]==valueZ)[0] 
        arr=copy(self.X_train[rows])
        #print 'arr is ',arr.shape
        #now we have the array for all thise values
        rows=np.where(arr[:,indexY]==valueY)
        #print 'rows is ',rows
        arr=copy(arr[rows])
        
        #now taking all training vectors with X
        
        rows=np.where(arr[:,indexX]==valueX)[0]
        #print 'rows is ',rows
        count=0
        if len(rows)>0:
            arr=copy(arr[rows])[0]
            count=arr.shape[0]
        else:
            count=0;
        
        #now we have that array 
        
        return  (count+1,float(count+1)/(self.X_train.shape[0]+2)) # adding a Beta Prior
           
    def createMaximumSpanningTree(self):
        
        self.pnode=np.zeros((1, self.mutalInformationMatrix.shape[0]))
        self.pdist=np.zeros((1, self.mutalInformationMatrix.shape[0]))     
        self.parents=np.zeros((1, self.mutalInformationMatrix.shape[0]))
        
        for i in range(0,self.pdist.shape[1]):
            self.pdist[0,i]=-10
            self.pnode[0,i]=i
            self.parents[0,i]=i
            
        newNode= self.pnode[0,-1]
        l1=list(reversed(range(0,)))
        print 'newNode is ',newNode
        self.graphMst=[]
        for m in reversed(range(0,self.mutalInformationMatrix.shape[0])):
            print 'm is ',m
            if m==0:
                continue
            maxDistance=-10
            maxIndex=0
            for i in range(0,m):
                thisEdge=self.mutalInformationMatrix[m,i] #since its a symmetric matrix
                if thisEdge > self.pdist[0,i] :
                    self.pdist[0,i]=thisEdge
                    self.parents[0,i]=newNode
                if maxDistance < self.pdist[0,i]:
                    maxDistance=self.pdist[0,i]
                    maxIndex=i
            print 'maxIndex is ',maxIndex,' and newNode is ',newNode,'\n\n'
            self.graphMst.append((int( self.parents[0,i]),int(self.pnode[0,maxIndex])))
            newNode=self.pnode[0,maxIndex]
            self.pnode[0,maxIndex]=self.pnode[0,m-1]
            self.pdist[0,maxIndex]=self.pdist[0,m-1]
            
    def plotGraph(self):
        
        G = nx.DiGraph()
        for i in range(0,self.mutalInformationMatrix.shape[0]+1):
            G.add_node(i)
        G.add_edges_from(self.graphMst)
        #ADD THE EDGE FOR THE LABEL NODE TOO
        for i in range(0,self.mutalInformationMatrix.shape[0]):
            G.add_edge(self.X_train.shape[1]-1,i)
        pos=nx.spring_layout(G)
        nx.draw_networkx_nodes(G,pos)
        nx.draw_networkx_labels(G,pos)
        nx.draw_networkx_edges(G,pos)
        plt.show()
        #A=A=nx.to_agraph(G)
        pos = graphviz_layout(G)
        #nx.draw(G,pos)
        #plt.show()
        write_dot(G,'BayesNet.dot')
        print 'so far so good'
                

    def calculateConditionalMutualInformation(self):
        #Now we have all the data we need to create the Mutual Information Array
        self.mutalInformationMatrix=np.zeros((self.X_train.shape[1]-1,self.X_train.shape[1]-1))
        #now we initialized the mutual Information matrix
        #Uptill now we have our starter Mutual Information Matrix and we have all the helper functions with us
        # we need to iterate through all the attributes and find the Conditional Mutual Information between all the variables
        for X in range(0,self.numAttributes-1):
            for Y in range(X+1,self.numAttributes):
                #we loop over all possible assignment of variables
                #we need to find I(X;Y|Z)=sum (x,y,z)P(x,y,zlog(p(x,y|z)/(p(x|z)p(y|z))))
                #we need to loop over all possible values of X ,Y and Z
                value=0.0
                for valueX in self.variableValues[X]:
                    for valueY in self.variableValues[Y]:
                        for valueZ in self.labels:
                       #     print 'X,Y = ',X,'=',valueX,',',Y,'=',valueY,' Z= ',valueZ
                            t1,temp=self.findProbability3Variables(X,valueX,Y,valueY,self.X_train.shape[1]-1,valueZ)
                            temp2=self.findConditionalProbability3Variables(X,valueX,Y,valueY,self.X_train.shape[1]-1,valueZ)
                            temp3=self.findConditionalProbability2Variables(X,valueX,self.X_train.shape[1]-1,valueZ)*self.findConditionalProbability2Variables(Y,valueY,self.X_train.shape[1]-1,valueZ)
                            #print 'temp2 is ',temp2,' temp3 is ',temp3
                            
                            value=temp*np.log2(temp2/temp3)
                            self.mutalInformationMatrix[X,Y]+=value
                self.mutalInformationMatrix[X,Y]=-1*self.mutalInformationMatrix[X,Y]
                self.mutalInformationMatrix[Y,X]=self.mutalInformationMatrix[X,Y]
                            
        print 'self.mutalInformationMatrix is ',self.mutalInformationMatrix
        
    def loadData(self,file1):
        f=open(file1,'r')
        l=[]
        for line in f:
            line=line.rstrip()
            chars=line.split(',')
            l.append(chars)
        
        arr=np.asarray(l)
        return arr


ob1=TAN('Chess (King-Rook vs. King-Pawn) Data Set.data')

        
