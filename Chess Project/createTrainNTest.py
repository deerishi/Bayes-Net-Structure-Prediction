import numpy as np
from copy import copy
from sklearn.utils import shuffle


class NaiveBayes:
    
    def __init__(self):
        pass
    
    def makethetas(self):
        labels=copy(self.train[:,-1])
        
        #Step1 make thea vector for class won
        rows=np.where(self.train[:,-1]=='won')[0]
        win=copy(self.train[rows])
        rows=np.where(self.train[:,-1]=='nowin')[0]
        nowin=copy(self.train[rows])
        #now we have the data to calculate thetas for the won and nowin senarios
        thetasWon=[]
        thetasNowin=[]
        #first define the labelling for the feature vectors since they are all characters right now
        #first we can find the class probabilities
        self.pwon=float(win.shape[0])/self.train.shape[0]
        self.pnowin=float(nowin.shape[0])/self.train.shape[0]
        for i in range(0,win.shape[1]-1):
            rows=np.where(win[:,i]=='1')[0]
            rows2=np.where(nowin[:,i]=='1')[0]
            #print 'rows is ',rows
            thetasWon.append(float(rows.shape[0]+1)/(win.shape[0]+2))
            thetasNowin.append(float(rows2.shape[0]+1)/(nowin.shape[0]+2))
            
        self.thetasWon=copy(np.asarray(thetasWon))
        self.thetasNowin=copy(np.asarray(thetasNowin))
        print ' self.thetasWon is ', self.thetasWon
        print 'self.thetasNowin=copy(np.array(thetasNowin)) is ',self.thetasNowin.shape    
    
    
    def checkAccuracy(self,predicted,goldset):
        predicted=predicted.tolist()
        goldset=goldset.tolist()
        correct=0
        
        
        for i in range(0,len(predicted)):
            if goldset[i]==predicted[i]:
                correct+=1
    
        return (float(correct)/len(predicted))*100
        
        
    def predict(self):
        
        goldset=self.test[:,-1]
        predicted=[]
        thetaWinLog=(self.thetasWon.reshape(38,1))
        One_thetaWinLog=(1-self.thetasWon).reshape(38,1)
        
        thetaNowin=(self.thetasNowin.reshape(38,1))
        One_thetaNoWin=(1-self.thetasNowin).reshape(38,1)
        
        for i in range(0,self.test.shape[0]):
            fv=copy(self.test[i,0:-1])
            fv=[int(i) for i in fv]
            fv=np.asarray(fv)
            fv=fv.reshape(1,-1)
            #now we have the feature vectors
            #now we can run our classfier
            print 'fv is ',np.dot(fv,np.log10(thetaWinLog)).shape
            pwin=np.dot(fv,np.log10(thetaWinLog))+np.dot(1-fv,np.log10(One_thetaWinLog))
            pwin=pwin[0][0]+np.log10(self.pwon)
            pnowin=np.dot(fv,np.log10(thetaNowin))+np.dot(1-fv,np.log10(One_thetaNoWin))
            pnowin=pnowin[0][0]+np.log10(self.pnowin)
            
            print 'pwin is ',pwin,' pnowin is ',(pnowin)
            if pwin > pnowin :
                predicted.append('won')
            else:
                predicted.append('nowin')
        
        res=self.checkAccuracy(np.asarray(predicted),goldset)
        print 'res is ',res
        
    
    
    def createTrainTest(self):
        data=np.load('Chess End Game Data/ChessArray.npy')
        labels=np.load('Chess End Game Data/ChessLabels.npy')
        allData=np.load('Chess End Game Data/ChessFull.npy')
        print 'allData is ',allData
        print 'allData is ',allData.shape
        allData=shuffle(allData, random_state=42)
        print 'allData now is ',allData
        #now we should split into tain and test_data
        train=copy(allData[0:2130])
        test=copy(allData[2130:])
        self.train=copy(train)
        self.test=copy(test)
        r1=np.where(train[:,-1]=='won')[0]
        r2=np.where(train[:,-1]=='nowin')[0]
        print 'r1 is ',r1.shape
        print 'r2 is ',r2.shape
        np.save('Chess End Game Data/Train/TrainWithLabels',train)
        np.save('Chess End Game Data/Test/TestWithLabels',test)
        self.makethetas()
        self.predict()
        
    
ob1=NaiveBayes()
ob1.createTrainTest()
