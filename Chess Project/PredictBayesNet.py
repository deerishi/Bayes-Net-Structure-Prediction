import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot
from sklearn.utils import shuffle


#we need to usea beta prior of ~Beta(2,2) since Z is a bernoulli rndom variable , so we need to add the halllucinated counts 
#simply add +2 to the count of P(Z) in the denominator and ad +1 in the numberator for eveything 
np.set_printoptions(threshold='nan')

class TAN:
    
    def __init__(self,file1):  
        self.variableValues={}
        self.X_train,self.X_test=self.loadData(file1)
        self.labels=set(self.X_train[:,-1])
        for i in range(0,self.X_train.shape[1]-1):
            s1=set(self.X_train[:,i])
            self.variableValues[i]=s1
        
        #print 'X_train is ',self.X_train
        print 'the possible variables values are ',self.variableValues
        #print 'labels are ',self.labels
        self.numAttributes=self.X_train.shape[1]-1 #we innore the labels
        print 'now saving x_train is ',self.X_train.shape
        print '\nnow saving x_test is ',self.X_test.shape
        np.save('ChessDataTrain',self.X_train)
        np.save('ChessDataTest',self.X_test)
        self.calculateConditionalMutualInformation()
        
        self.createMaximumSpanningTree()
        print 'the GRAPH mst is ',self.graphMst,'\n'
       
        self.plotGraph()
        
        print 'now testing '
        self.test()
    
    
    def findConditionalProbability3Variables(self,indexX,valueX,indexY,valueY,indexZ,valueZ):
        
        #Here we find P(X,Y|Z)= #count(X&Y&Z)/#count(Z)
        
        pxyz=self.findProbability3Variables(indexX,valueX,indexY,valueY,indexZ,valueZ)
        
        rows=np.where(self.X_train[:,indexZ]==valueZ)[0]
        arr=copy(self.X_train[rows])
        
        pz=float(arr.shape[0]+1)/(self.X_train.shape[0]+2)
        
        return float(pxyz)/(pz) # accounting for beta distibution
        
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
        
    def findProbability2Variables(self,indexX,valueX,indexZ,valueZ):
    
        rows=np.where(self.X_train[:,indexZ]==valueZ)[0]
        arr=copy(self.X_train[rows])
        
        rows=np.where(arr[:,indexX]==valueX)[0]
        arr=copy(arr[rows])
        
        numerator=float(arr.shape[0])
        denominator=self.X_train.shape[0]    
        
        return (numerator+1)/(denominator+2)
        
        
        
    def findProbability3Variables(self,indexX,valueX,indexY,valueY,indexZ,valueZ):
        #we find P(X,Y,Z)
        #The maximum lkelyhood estimate is #(x&y&Z)/totalnumber of datapoints
        #print 'val'
        rows=np.where(self.X_train[:,indexZ]==valueZ)[0]
        arr=copy(self.X_train[rows])
        
        rows=np.where(arr[:,indexY]==valueY)[0]
        arr=copy(arr[rows])
        
        rows=np.where(arr[:,indexX]==valueX)[0]
        count=0
        if len(rows)>0:
            arr=copy(arr[rows])
            count=arr.shape[0]
        
        return  float(count+1)/(self.X_train.shape[0]+2) # adding a Beta Prior
           
    def createMaximumSpanningTree(self):
        
        self.pnode=np.zeros((1, self.mutalInformationMatrix.shape[0]))
        self.pdist=np.zeros((1, self.mutalInformationMatrix.shape[0]))     
        self.parents=np.zeros((1, self.mutalInformationMatrix.shape[0]))
        
        self.nodeImmediateParents={} #for storing the node and its immediate parent
        
        
        for i in range(0,self.pdist.shape[1]):
            self.pdist[0,i]=-10
            self.pnode[0,i]=i
            self.parents[0,i]=i
            
        newNode= self.pnode[0,-1]

        print 'newNode is ',newNode
        self.nodeImmediateParents[int(newNode)]=-1
        self.graphMst=[]
        for m in reversed(range(0,self.mutalInformationMatrix.shape[0])):
            print 'm is ',m
            if m==0:
                continue
            maxDistance=-10
            maxIndex=0
            for i in range(0,m):
                thisEdge=self.mutalInformationMatrix[newNode,self.pnode[0,i]] #since its a symmetric matrix
                if thisEdge > self.pdist[0,i] :
                    self.pdist[0,i]=thisEdge
                    self.parents[0,i]=newNode
                if maxDistance < self.pdist[0,i]:
                    maxDistance=self.pdist[0,i]
                    maxIndex=i
            
            self.graphMst.append((int( self.parents[0,maxIndex]),int(self.pnode[0,maxIndex])))
            
            self.nodeImmediateParents[int(self.pnode[0,maxIndex])]=int( self.parents[0,maxIndex])
            print 'adding ',int( self.parents[0,maxIndex]),' -->  ',int(self.pnode[0,maxIndex])
            newNode=self.pnode[0,maxIndex]
            self.pnode[0,maxIndex]=self.pnode[0,m-1]
            self.pdist[0,maxIndex]=self.pdist[0,m-1]
        
        print 'the immediate parents for the node are ',self.nodeImmediateParents
    
    def renameNodes(self):
        names={}
        for i in range(0,self.X_train.shape[1]-1):
            nameStr=str(i)+" = {"
            for node in self.variableValues[i]:
                nameStr+=str(node)+","
            
            nameStr=nameStr[:-1]
            nameStr+="}"
            names[i]=nameStr
        nameStr=str(self.X_train.shape[1]-1)+" = {"  
        for label in self.labels:
            nameStr+=str(label)+","
        
        nameStr=nameStr[:-1]
        nameStr+="}"
        names[self.X_train.shape[1]-1]=nameStr
        print 'names is ',names
        self.G=nx.relabel_nodes(self.G,names)
        #self.plotGraph()
      
    def findZ(self,indexZ,valueZ):
    
        rows=np.where(self.X_train[:,indexZ]==valueZ)[0]
        arr=copy(self.X_train[rows])
        
        return float(arr.shape[0])/self.X_train.shape[0]
          
     
    def inferenceOnTan(self,dataTest):
        
        #since we have the dictionary of immediate parents and we know every node will have the label as its parent too
        #we have self. test
        
        dataTest=copy(dataTest)
        dataTest=dataTest.reshape(1,-1)
        label=dataTest[0,-1]
        productWon=1.0
        productNowin=1
        for i in range(0,dataTest.shape[1]-1):
            parent=self.nodeImmediateParents[i]
            if self.nodeImmediateParents[i]!=-1:
                productWon=productWon*self.findProbability3Variables(i,dataTest[0,i],parent,dataTest[0,parent],dataTest.shape[1]-1,'won') 
                productWon=productWon/self.findProbability2Variables(parent,dataTest[0,parent],dataTest.shape[1]-1,'won')
                
                
                productNowin=productNowin*self.findProbability3Variables(i,dataTest[0,i],parent,dataTest[0,parent],dataTest.shape[1]-1,'nowin')
                
                productNowin=productNowin/self.findProbability2Variables(parent,dataTest[0,parent],dataTest.shape[1]-1,'nowin')
    
            else:

                productWon=productWon*self.findConditionalProbability2Variables(i,dataTest[0,i],dataTest.shape[1]-1,'won')
                productNowin=productNowin*self.findConditionalProbability2Variables(i,dataTest[0,i],dataTest.shape[1]-1,'nowin')
                
        productWon=productWon * self.findZ(dataTest.shape[1]-1,'won')
        productNowin=productNowin *  self.findZ(dataTest.shape[1]-1,'nowin')        
        print 'productwonis is ',productWon,' productNowin is ',productNowin,'\n'
        
        if productWon > productNowin:
            return 'won'
        else:
            return 'nowin'
            
    
    def test(self):
        
        predictions=[]
        for i in range(0,self.X_test.shape[0]):
            predictions.append(self.inferenceOnTan(self.X_test[i,:]))
        
        goldset=copy(self.X_test[:,-1])
        predictions=np.asarray(predictions)
        
        print 'the accuracy is ',self.checkAccuracy(predictions,goldset),'\n'
                
    
    def checkAccuracy(self,predicted,goldset):
        predicted=predicted.tolist()
        goldset=goldset.tolist()
        correct=0
        for i in range(0,len(predicted)):
            if goldset[i]==predicted[i]:
                correct+=1
        
        return (float(correct)/len(predicted))*100
                            
                
    def countDifferentLabels(self,arr):
        
        w=0
        nw=0
        for i in range(0, arr.shape[0]):
            if arr[i,-1]=='won':
                w+=1
            else:
                nw+=1
        
        print 'number of wons are ',w
        print 'number of no wins are ',nw
        
             
            
    def plotGraph(self):
        
        self.G = nx.DiGraph()
        for i in range(0,self.mutalInformationMatrix.shape[0]+1):
            self.G.add_node(i)
        self.G.add_edges_from(self.graphMst)
        #ADD THE EDGE FOR THE LABEL NODE TOO
        for i in range(0,self.mutalInformationMatrix.shape[0]):
            self.G.add_edge(self.X_train.shape[1]-1,i)
        
        self.renameNodes()
        pos=nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G,pos)
        nx.draw_networkx_labels(self.G,pos)
        nx.draw_networkx_edges(self.G,pos)
        plt.show()
        #A=A=nx.to_agraph(G)
        pos = graphviz_layout(self.G)
        #nx.draw(G,pos)
        #plt.show()
        write_dot(self.G,'BayesNet4.dot')
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
                            temp=self.findProbability3Variables(X,valueX,Y,valueY,self.X_train.shape[1]-1,valueZ)
                            temp2=self.findConditionalProbability3Variables(X,valueX,Y,valueY,self.X_train.shape[1]-1,valueZ)
                            temp3=self.findConditionalProbability2Variables(X,valueX,self.X_train.shape[1]-1,valueZ)*self.findConditionalProbability2Variables(Y,valueY,self.X_train.shape[1]-1,valueZ)
                            #print 'temp2 is ',temp2,' temp3 is ',temp3
                            
                            value=temp*np.log2(temp2/temp3)
                            self.mutalInformationMatrix[X,Y]+=value
                self.mutalInformationMatrix[X,Y]=self.mutalInformationMatrix[X,Y]
                self.mutalInformationMatrix[Y,X]=self.mutalInformationMatrix[X,Y]
                            
        print 'self.mutalInformationMatrix[0][1] is ',self.mutalInformationMatrix[10][14]
        
    def loadData(self,file1):
        f=open(file1,'r')
        l=[]
        for line in f:
            line=line.rstrip()
            chars=line.split(',')
            l.append(chars)
        
        arr=np.asarray(l)
        #we will only need to shuffle here
        arr=shuffle(arr,random_state=42)
        #print 'arr is ',arr
        self.X_train=copy(arr[0:2130])
        self.X_test=copy(arr[2130:])
        
        print 'for training set \n'
        self.countDifferentLabels(self.X_train)
        
        print 'for testing set \n'
        self.countDifferentLabels(self.X_test)
               
        return (self.X_train,self.X_test)

ob1=TAN('Chess (King-Rook vs. King-Pawn) Data Set.data')      
