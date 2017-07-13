import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../util'))
from Utils import cost,init_weight_and_bias,y2indicator,error_rate,get_data,get_test_data
from ANNHelper import relu,Forward,derivative_with_1,derivative_with_2,derivative_with_b2,derivative_with_b1
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle

#Fetching the training data
print("Fetching Training Data...")
X,T = get_data("../data/train.csv")

#one hot encoding for the trainging image labels
T = y2indicator(T)
XTrain = X
YTrain = T
print("Done Fetching Data...")

#N = number of samples and D = Number of features
N,D = XTrain.shape
#M hidden states
M=100
#K is size of one hot encoded output 
K=T.shape[1]

#intializing weights and bias
W1,b1 = init_weight_and_bias(D,M)
W2,b2 = init_weight_and_bias(M,K)

#regularization rate
reg = 0.01
#learning rate
learning_rate = 0.0001

#batch size for batch gradient descent
batch_sz = int(500)

#performing RMS Decay with a decay of 0.999
cache_W2=0
cache_b2=0
cache_W1=0
cache_b1=0
eps = 0.0000000001
decay_rate=0.999
LL=[]
num_batches = int(N/batch_sz)

for i in range(700):
    for each_batch in range(num_batches):
        XBatch = XTrain[each_batch*batch_sz:(each_batch*batch_sz+batch_sz)]
        YBatch = YTrain[each_batch*batch_sz:(each_batch*batch_sz+batch_sz)]
        output,hidden = Forward(XBatch,W1,b1,W2,b2)
        #performing decay for all weights
        gW2 = (derivative_with_2(YBatch,output,hidden)+reg*W2)
        cache_W2 = decay_rate*cache_W2+(1-decay_rate)*gW2*gW2
        W2 = W2 + learning_rate*gW2/(np.sqrt(cache_W2)+eps)

        gb2 = (derivative_with_b2(YBatch,output)+reg*b2)
        cache_b2 = decay_rate*cache_b2+(1-decay_rate)*gb2*gb2
        b2 = b2 + learning_rate*gb2/(np.sqrt(cache_b2)+eps)

        gW1 = (derivative_with_1(XBatch,YBatch,output,hidden,W2)+reg*W1)
        cache_W1 = decay_rate*cache_W1+(1-decay_rate)*gW1*gW1
        W1 = W1 + learning_rate*gW1/(np.sqrt(cache_W1)+eps)

        gb1 = (derivative_with_b1(YBatch,output,hidden,W2)+reg*b1)
        cache_b1 = decay_rate*cache_b1+(1-decay_rate)*gb1*gb1
        b1 = b1 + learning_rate*gb1/(np.sqrt(cache_b1)+eps)
        if each_batch % 100 == 0:
            ll = cost(YBatch,output)
            LL.append(ll*-1)
            err = error_rate(YBatch,output)
            print ("Error is "+str(err)+" for i="+str(i))

#performa plot of cost vs time
plt.plot(LL, label='cost')
plt.legend()
plt.show()

#loading validation dataset
XTest = get_test_data("C:\\Users\\prdha\\Desktop\\ANN\\test.csv")

#performing prediction on validation set
py,_ = Forward(XTest,W1,b1,W2,b2)

#calculating the output image label from one hot encoded output
#err = error_rate(YBatch,output)
#print("Test Error Rate : "+str(err))
py = np.argmax(py,axis=1)

NTest = XTest.shape[0]
f = open("C:\\Users\\prdha\\Desktop\\ANN\\Result.csv","w")
f.write("ImageId,Label\n")
for n in range(NTest):
    f.write(str(n+1)+","+str(py[n])+"\n")
f.close()

    

