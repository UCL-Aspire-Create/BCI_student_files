
from ctypes.wintypes import BOOLEAN
from multiprocessing import Value
from typing import Iterable
from xmlrpc.client import boolean
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import random

import scipy as sci
from scipy import signal, misc

import sklearn
from sklearn import impute
import warnings


class Data_processor():

    #warnings.filterwarnings('ignore')
    notch_filter = None
    band_filter = None

    def create_notch_filter(self,Fs:int)->None:

        """creates the data processors notch filter. 
       Parameters:
       Fs: the frequncey you want to filter out
       """

        nyq = int(Fs/2)

        freq = [0, 49.9/nyq, 50/nyq, 50.1/nyq, (nyq-1)/nyq, 1]

        gain = [1,1,0,1,1,0]

        notch = signal.firwin2(nyq,freq,gain)

        self.notch_filter = notch

   
    def Load_BCI_data(self,csv_files: Iterable[str]) -> pd.DataFrame:

       """Loads all CSV files into a dataframe
       Parameters:
       csv_files: A string array of file paths for the EEG CSV recordings 
       """

       BCI_df = pd.read_csv(csv_files[0].strip("\n"))
       print(BCI_df.shape)
       #print(BCI_df.head())
    
       if len(csv_files) > 1:
          for i in range(len(csv_files)-1):
              tempdf = pd.read_csv(csv_files[i+1].strip("\n"))
              #print(tempdf.head())
              BCI_df = BCI_df.append(tempdf,ignore_index=True)
              print(BCI_df.shape)
          return BCI_df
       else:
          return BCI_df

    def convert_to_freq_2D(self,data:np.ndarray,UseNotch:bool)->np.ndarray:

        """convert a 2D numpy ndarray (aimed to be a single epoch) into frequency domain
        Parameters:
        data: 2D array for main numpy ndarray representing a single epoch.
        Axis 0 = Time
        Axis 1 = Channels

        UseNotch: Boolean representing if the data will be filted with a notch filter 

        Returns 2D numpy ndarray data [Time,Channels]
        """

        if  UseNotch == False:
            tempf = signal.filtfilt(self.band_filter,1,data,axis=0,padtype = None)
            return np.square(self.transform(tempf)) 

        else:
            tempf = signal.filtfilt(self.notch_filter,1,data,axis=0,padtype = None)
            tempf = signal.filtfilt(self.band_filter,1,data,axis=0,padtype = None)
            tempf = np.square
            return np.square(self.transform(tempf))


    def convert_to_freq(self,data:np.ndarray,UseNotch:bool)->np.ndarray:
       """convert a 3D numpy ndarray into frequncey domaain (aimed to be a set of epochs)
      Parameters:
      data[Time,Channel,Epoch]: 3D array representing multiple epochs
     
      UseNotch: Boolean representing if the data will be filted with a notch filter 

      Returns 3D numpy  ndarray data[Epoch,time,channels]
      """

       if  UseNotch == False:
           tempf = signal.filtfilt(self.band_filter,1,data[:,:,0], axis =0,padtype = None)
           output = self.transform(tempf)
    
           for i in range(data.shape[2] -1):
        
              tempf = signal.filtfilt(self.band_filter,1,data[:,:,(i+1)], axis =0,padtype = None)
              tempt = self.transform(tempf)
              psd = np.square(tempt)
              output = np.dstack((output,psd))
          
       else:
    
         tempf = signal.filtfilt(self.notch_filter,1,data[:,:,0], axis =0,padtype = None)
         tempf = signal.filtfilt(self.band_filter,1,tempf, axis =0,padtype = None)
         output = self.transform(tempf)
    
         for i in range(data.shape[2] -1):
        
            tempf = signal.filtfilt(self.notch_filter,1,data[:,:,(i+1)], axis =0,padtype = None)
            tempf = signal.filtfilt(self.band_filter,1,tempf, axis =0,padtype = None)
            tempt = self.transform(tempf)
            psd = np.square(tempt)
            output = np.dstack((output,psd))
    
       output = np.swapaxes(output,0,2)
       output = np.swapaxes(output,1,2)

       #print(output.shape)
        
        
       return output

    def transform(self,epoch:np.ndarray)->np.ndarray:

       """Performs a Fast fourier transform on a 2D array along Axis 0
       
       Parameters:
       Epoch[time,channel]: a 2D numpy ndarray representing one epoch
       (If you would like to perform it on one channel, parse in a 2D array with the size (n,1) )

       Returns: 2D np.ndarray 
       
       """
    
       L = len(epoch)
    
       Y = np.fft.fft(epoch, L, axis = 0)
    
    
       P2 = abs(Y/L)
       P1 = P2[1:(int(L/2))+1,:]
       P1 [2:len(P1)-2,:] = 2 * P1[2:len(P1)-2,:]
    
       return P1

    def createBPF(self,Fs:int,upperBound:int)->None:

        """Creates the band pass filter for the for the data processor
        
        Parameters:
        upperBound: the int value the upperbound of the band pass filter
        
        Returns: None
        """

        nyq = int(Fs/2)

        freq = [0, 7/nyq, 7.1/nyq, upperBound/nyq, (upperBound+0.1)/nyq, 1]

        gain = [0,0,1,1,0,0]

        bpf = signal.firwin2(nyq,freq,gain)

        self.band_filter = bpf


    def extract_epoch(self,data:np.ndarray,stim_index:int,epoch_length:float,Fs:int)->np.ndarray:

       """Extracts an epoch of a specified lenght from a larger recordong

       Parameters:
       data[Time,channels]: 2D numpy ndarray representing the whole data set
       stim_index: the position alonng the 0 axis indincation the time when the stimulation started
       epoch_lenth: float indecating the length of th epoch in seconds
       Fs: int representing the sampling frequnecy of the EEG recording

       Returns: 2D numpy ndarray
       
       """
    
       #discard = 2 * Fs
       endval = int((epoch_length * Fs) + stim_index)
    
       edited_epoch = data[stim_index:endval,:]
       #edited_epoch = edited_epoch[discard:,:]  
    
       #print (edited_epoch.size)
       return edited_epoch



    def LoadRouteFile(self,ifMulti, Route_txt):

       """Obsolete function used for matching manual triggers with a list of positions"""
    
       classes = np.array([])
       if ifMulti == 1:
        
          for file in Route_txt:
               f = open(file,"r")
               tempclasses = f.readline()
               tempclasses = tempclasses.split(',')
               tempclasses = np.array(tempclasses)
               tempclasses = tempclasses[1:]
               classes = np.append(classes,tempclasses)
          return classes
        
       else:
        
          f = open(Route_txt,"r")
        
          while True:
              tempclasses = f.readline()
              if not tempclasses:
                 break
              else:
                tempclasses = tempclasses.split(',')
                tempclasses = np.array(tempclasses)
                tempclasses = tempclasses[1:]
                classes = np.append(classes,tempclasses)
          return classes

    def flatten_data(self,data:np.ndarray)->np.ndarray:

        """Fatterns 3D [epoch,time,channels] data into 2D [example:data] data set for machine learning data set

        Parameters:
        data[epoch,time,channels]: 3D array representing epochs of a certain type or classification


        Returns: 2D numpy ndarray
        """

    
        num = data.shape[0]
    
        output = np.zeros((num,data[0,:,:].size))
    
        for i in range(num):
            output[i,:] = data[i,:,:].flatten()
        
        return output
    

      

    def create_testset(self,x:np.ndarray,y:np.ndarray)->None:
        testsetx = pd.DataFrame(x)
        
        testsety = pd.DataFrame(y)
        
        
        testsetx.to_csv('./testsetx.csv')
        testsety.to_csv('./testsety.csv')


    def Divide_data(self,data:np.ndarray,epoch_Length:float,Fs:int)-> np.ndarray:

        """Divides a large recording into multiple smaller epochs

        Parameters:
        data[time,channels]: 2D numpy ndarray representing a recording of EEG data
        epoch_Length: float representing the lenth of the epoch you want extracted in seconds
        Fs: int representing the sampling frequency of the data

        Returns: 3D numpy ndarray [time, channel, epoch]
        
        
        """

        
        jump = int(Fs/2)
        epoch_size = int(Fs * epoch_Length)
        epoch_size = 256
        output = np.zeros((epoch_size,data.shape[1]))

        pointer = 0
 
        while True:
            if pointer + epoch_size >= data.shape[0]:
                break
            
            else:
                output=np.dstack((output,data[pointer:(pointer + epoch_size),:]))
                pointer += jump
                pointer = int(pointer)
        
        return output
        
        
    def Process_Chunk(self,data:np.ndarray,UseNotch:bool,epoch_length:float,Fs:int)->np.ndarray:

        """Takes recording of data in the same class and processes it into frequncey based epochs 
        Parameters:

        data[time,channels]: 2D numpy ndarray representing a recording of EEG data
        UseNotch: Boolean value representing weatehr a notch filter should be used
        epoch_Length: float representing the lenth of the epoch you want extracted in seconds
        Fs: int representing the sampling frequency of the data 

        
        Returns: 3D numpy ndarray [epoch,time,channel]
        """


        data = self.Divide_data(data,epoch_length,Fs)
        data = self.convert_to_freq(data,UseNotch)
        return data


    def Create_logistic_dataset(self,data_set_array:Iterable[np.ndarray], designated_class:int = 1):

        """Creates a data set, training and lables for logistic regression and other types of classifier

        Parameters:
        data_set_array[numpy ndarray]: An array of 3D numpy ndarrays [epoch,time,channel] containg data of different classifications  
        
        designated_class: int used when only one data set is in the array. sets the label to that class

        Returns: 2 2D numpy ndarrays
        
        """

        classes = len(data_set_array)
    
        if classes == 1:
            data = self.flatten_data(data_set_array[0])
            y = np.ones(data.shape[0]) * designated_class
        
    
    
        else:
            data = self.flatten_data(data_set_array[0])
            y = np.ones(data.shape[0])
        
            for i in range(classes -1):
                temp = self.flatten_data(data_set_array[i+1])
                data = np.append(data,temp,axis = 0)
                temp_y = np.ones(temp.shape[0]) * (i+2)
                y = np.append(y,temp_y)
            
        return data, y

    def Create_Numbered_dataset(self,data_set_array:np.ndarray, designated_class:int = 1)->np.ndarray:
        classes = len(data_set_array)


        """Creates a data set, training and lables for Nural networks

        Parameters:
        data_set_array[numpy ndarray]: An array of 3D numpy ndarrays [epoch,time,channel] containg data of different classifications  
        
        designated_class: int used when only one data set is in the array. sets the label to that class

        Returns: 2 2D numpy ndarrays
                Data[epoch,data]
                y[epoch,class]
        
        """
    
        if classes == 1:
            data = self.flatten_data(data_set_array[0])
            y = np.ones(data.shape[0]) * designated_class
        
    
    
        else:
            data = self.flatten_data(data_set_array[0])
            y = np.ones(data.shape[0])
            
            print(data.shape)
        

            for i in range(classes -1):
                print(data_set_array[i+1].shape)
                temp = self.flatten_data(data_set_array[i+1])
                print(temp.shape)
                data = np.append(data,temp,axis = 0)
                temp_y = np.ones(temp.shape[0]) * (i+2)
                y = np.append(y,temp_y)
            
        return data, y

    def create_test_set2(self,X:np.ndarray,y:np.ndarray)->np.ndarray:

        """Creates randomly selected test set from data  and labels

        Parameters:
        X[epoch,data]: 2D numpy ndarray contained the falttened data of each epoch and a  
        y[lables]: 1D numpy ndarray containg the labels

        Returns: X[epoch, data], X_test[epoch, data] 2D numpy ndarray containing epochs and their data  
                 y[lables], Y_Test[lables] 1D numpy ndarray contains 
        
        """


        #print(X.shape)
        #print(y.shape)
        y = y[:,np.newaxis]
        data = np.append(X,y,axis = 1)
        print(data.shape)
        rng = np.random.default_rng()
        data =  rng.permutation(data,axis = 0)
    
        percentage = int(data.shape[0] * 0.2)
    
        test_set = np.zeros(data.shape[1])[np.newaxis,:]
    
        for i in range(percentage):
            index = random.randint(0,data.shape[0]-1)
            test_set = np.append(test_set,data[index,:][np.newaxis,:],axis = 0)
            data = np.delete(data,index,0)
        
        print(test_set.shape)
        test_set = test_set[1:,:]
    
        Y_test = test_set[:,-1]
        X_test = np.delete(test_set,-1,1)
    
        y = data[:,-1]
        X = np.delete(data,-1,1)
    
        return X,y,X_test,Y_test



    def convert_to_bins(self,data:np.ndarray,bin_size:int)->np.ndarray:
        numbins = int(data.shape[1]/bin_size)
        temp_array = np.zeros(numbins)




        output = np.zeros((numbins,data.shape[2]))
        holder = np.zeros((numbins,1))

        #bins = np.array_split(data[i,:,j],numbins)
        #print(bins)



        for i in range(data.shape[0]):

            for j in range(data.shape[2]):

                bins = np.array_split(data[i,:,j],numbins)


                for a in range(len(bins)):
                    temp_array[a] = np.mean(bins[a])

                temp_array= temp_array[:,np.newaxis]
                holder = np.append(holder,temp_array, axis = 1)
                #print(holder.shape)
                temp_array = np.zeros(numbins)

            output = np.dstack((output,holder[:,1:]))
            holder = np.zeros((numbins,1))

    
        output = np.swapaxes(output,0,2)
        output = np.swapaxes(output,1,2)
        output = output[1:,:,:]
        return output


    def Convert_to_bins_single(self,data:np.ndarray,bin_size:int)->np.ndarray:

        """converts the data set into bins of defines size
        
        Parameters:
        data[freq,channels]: 2D numpy ndarray representing a recording of EEG data
        bin_size: int number of values per bin

        Returns: 
        2D data array [freq,channel]
        
        """




        numbins = int(data.shape[0]/bin_size)
        temp_array = np.zeros(numbins)

        holder = np.zeros((numbins,1))

        for i in range(data.shape[1]):
            
            bins = np.array_split(data[:,i],numbins)

            for a in range(len(bins)):
                temp_array[a] = np.mean(bins[a])

            temp_array= temp_array[:,np.newaxis]
            holder = np.append(holder,temp_array, axis = 1)
            #print(holder.shape)
            temp_array = np.zeros(numbins)

        return holder[:,1:]



    

    def Create_NN_val_and_test(self,X:np.ndarray,y:np.ndarray)->np.ndarray:
        
        """Creates randomly selected test set and validation set from data  and labels

        Parameters:
        X[epoch,data]: 2D numpy ndarray contained the falttened data of each epoch and a  
        y[lables]: 1D numpy ndarray containg the labels

        Returns: X[epoch, data], X_val[epoch, data], X_test[epoch, data]  2D numpy ndarray containing epochs and their data  
                 y[lables], Y_val[lables] ,Y_Test[lables] 1D numpy ndarray contains 
        
        """

        y = y[:,np.newaxis]
        data = np.append(X,y,axis = 1)
        print(data.shape)
        rng = np.random.default_rng()
        data =  rng.permutation(data,axis = 0)
    
        percentage = int(data.shape[0] * 0.2)
    
        test_set = np.zeros(data.shape[1])[np.newaxis,:]
        val_set = np.zeros(data.shape[1])[np.newaxis,:]
    
        for i in range(percentage):
            index = random.randint(0,data.shape[0]-1)
            test_set = np.append(test_set,data[index,:][np.newaxis,:],axis = 0)
            data = np.delete(data,index,0)


        print(test_set.shape)
        test_set = test_set[1:,:]
    
        Y_test = test_set[:,-1]
        X_test = np.delete(test_set,-1,1)

        for i in range(percentage):
            index = random.randint(0,data.shape[0]-1)
            val_set = np.append(test_set,data[index,:][np.newaxis,:],axis = 0)
            data = np.delete(data,index,0)

        Y_val = val_set[:,-1]
        X_val = np.delete(val_set,-1,1)


        y = data[:,-1]
        X = np.delete(data,-1,1)

        return X,y,X_val,Y_val,X_test,Y_test

        

    def convert_y_for_single_NN(self,Y:np.ndarray,Value_array:Iterable[int])->np.ndarray:

        """Converts number lables multiple epochs for NN with 1 node in output layer
        
        Parameters: 
        Y: 1D numpy ndarray representing the lables
        Value_array: array representing which lable you want to be true of face
               example ( convert_y_for_single_NN([1,2,4,2],[1,0,0,1]) would equal = [1,0,1,0]
                         convert_y_for_single_NN([3,3,3,2],[0,0,1,0]) would equal = [1,1,1,0]
               )

        Returns: 1D numpy array

        """

        y = Y

        for i in range(y.shape[0]):
            if Value_array[int(Y[i]) - 1] == 1:
                y[i] = 1
            else:
                y[i] = 0

        return y

    def convert_y_for_NN_2(self,y:np.ndarray,size:int,value_array:Iterable[int])->np.ndarray:

        """Converts labels into NN format for NN that have multiple nodes on the output layer.

        Parameters:
        y: 1D numpy ndarray representing the lables
        size: int number of nodes on output layer
        value_array: int array representing which labels are equal which class.
                  example(
                  convert_y_for_NN_2([1,2,3,2,4] , 2 , [1,1,2,0]) would equal =
                  = [[1,0],
                     [1,0],
                     [0,1],
                     [1,0],
                     [0,0]] 
                  )

        Returns: 2D numpy ndarray
        
        """

        #print(len(value_array))
        #print(y.shape)

        new_y = np.zeros(size)[np.newaxis,:]
        
        for i in range(y.shape[0]):
            temp = np.zeros(size)[np.newaxis,:]
            
            if value_array[int(y[i])-1] != 0:
                temp[0,
                    value_array[
                    int(y[i])-1]-1] = 1

            new_y = np.append(new_y,temp,axis = 0)

        return new_y[1:,:]

    def save_subset_set(self,name:str,X:np.ndarray,Y:np.ndarray)->None:

        """Saves the passed data as test set CSV in root folder of the program

        Parameters:
        Name: string name of the file
        X: 2D numpy ndarray array containg flatterned Data
        Y:  numpy ndarray array containg either number or NN converted labels
       
        Returns: None
        """

        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(Y)

        X_df.to_csv(('./'+name+'_Xtest.csv' ))
        y_df.to_csv(('./'+name+'_Ytest.csv' ))
        print("Saved test Set")





           

    




