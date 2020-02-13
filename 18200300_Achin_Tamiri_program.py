#!/usr/bin/env python
# coding: utf-8

# # STUDENT NAME :ACHIN TAMIRI
# # STUDENT ID        :18200300
# 
# # Connectionist Computing Programming Assignment: Building Multi Layer Perceptron.

# In[191]:


#Import all The Required Libraries
import numpy as np
from numpy import  exp


# #### Function to generate random weights for the Neural Network Model

# In[192]:


def GenerateWeights(nodes,input_size) :
    weights =np.random.random((nodes+1,input_size))
    return weights


# #### Function to add bias in addition to the  nodes when given as input to the next layer 

# In[193]:


def bias(ip,examples_no):
    biass = np.ones(examples_no)#Generate bias column with 1's
    inputs_bias_result = np.c_[biass, ip] #concatenate bias column generated with the input array
    return inputs_bias_result


# #### Function for Activation Function Sigmoid

# In[194]:


def sigmoid(bias,LayerWeights):
    layerout = np.dot(bias,LayerWeights)#Perform ,Multiply input node values,add bias with the layerweights
    return  1 / (1 + exp(-layerout))# calculate sigmoid activation function from previous result
     


# #### Function which calculates and returns sigmoid derivative

# In[195]:


def sigDerivative(derivate):
    D = derivate*(1-derivate)#calculate derivate for backpropagation
    return D


# #### Function which returns node information like no of input node,no of output nodes,Total number of examples

# In[196]:


def node_info(xx,yy):
    num_input_node = xx.shape[1] # It gives number of input nodes
    num_output_node = yy.shape[1]# It gives number of input nodes
    total_no_examples = xx.shape[0]# It gives number of total examples to be trained
    return num_input_node,num_output_node,total_no_examples


# #### Function to Train Model ,Perform Forward and backword propagation and return best optimized weights

# In[197]:


def TrainModel(InputLayerWeights,hiddenlayerweights,ip_with_bias,output,epoch,Acceptance,total_no_examples):
    #FORWARD PROPAGATION
    for i in range(epoch):
        InputLayeroutput = sigmoid(ip_with_bias,InputLayerWeights)#call function sigmoid and pass input example with bias and InputLayerWeights.The output from first layer is applied sigmoid activation function and stored in InputLayeroutput
        Hidden_LayerIP = bias(InputLayeroutput,total_no_examples)#Add bias to the output from input layer and generate second hidden layer Hidden_LayerIP
        output_predicted = sigmoid(Hidden_LayerIP,hiddenlayerweights)#Call function sigmoid and pass input as output from first layer,perform activation and final output is the third final layer stored in output_predicted
        error_output = output-output_predicted # Calculate the difference between the actual defined output and generated output by forward propagation and store it in error_output
        erormax = max(abs(error_output))# select the maximun deviation from actual output in erormax
        if abs(erormax) < Acceptance:# define a particular acceptance rate say 0.1 minimun difference
           print ("Final Optimized weights obtained at epoch",i)# if deviation is less than 0.1 from from actual output reture These best optimised weights for first and second layer. 
           return hiddenlayerweights,InputLayerWeights                  
        delta_pred_out = error_output*sigDerivative(output_predicted)# IF NOT :- get the delta difference from the deviation from actual output and from derivative of the prdicted optuput from this train model for final layer
        error_hidden = np.dot(delta_pred_out,hiddenlayerweights.T)#Multiply it with second hidden layer weights and update the errors for hidden layer weights
        delta_hidden = error_hidden*sigDerivative(Hidden_LayerIP)# similarly find delta for the previous layer
        delta_hidden = np.delete(delta_hidden,0,axis=1) 
        # LEARNING RATE DEFINED AS : 0.1
        hiddenlayerweights+=(np.dot(Hidden_LayerIP.T,delta_pred_out))*0.1 #Update weights for Hidden layer to output layer by multiplying delta with hidden layer weights
        InputLayerWeights+=(np.dot(ip_with_bias.T,delta_hidden))*0.1 # Update the weights for input layer to second layer similaryly
        # Repeat the process till the maximun deviation is less than the defined acceptance for the range of epoch


# ## Predict Function which return the final predicted values based on the best optimized weights calculated in the trained model

# In[198]:


def PredictTrainFinalModel(X, w1, w2,total_no_examples):
  first_output = sigmoid(X,w1) 
  second_input = bias(first_output,total_no_examples)
  final_output = sigmoid(second_input,w2)
  return final_output  


# ## Task 1. XOR Implemention with 4 examples,2inputs,2hidden nodes,1Output node

# In[199]:


inputs=np.array([[0,0],[0,1],[1,0],[1,1]])# Define 4 examples as input for XOR operation with 2 inputs per example
output=np.array([[0,1,1,0]]).T  # output expected is 0,1,1,0 based on XOR inputs


# #### Call function node_info to get  no of input nodes,no of output nodes,Total number of examples which be used for generating weight for layers 

# In[200]:


no_of_ip_nodes,no_of_op_nodes,total_no_examples=node_info(inputs,output)
print("Inputs per node :",no_of_ip_nodes,"Output per nodes: ",no_of_op_nodes,"Total Examples",total_no_examples)


# #### Call function GenerateWeights to create First layer,Second(hidden layer), in the model 

# In[201]:


InputLayerWeights = GenerateWeights(no_of_ip_nodes,no_of_ip_nodes)
hiddenlayerweights = GenerateWeights(no_of_ip_nodes,no_of_op_nodes)
ip_with_bias= bias(inputs,total_no_examples)
print("First Input Layer weights \n",InputLayerWeights,"\n Second hidden Layer weights: \n",hiddenlayerweights)


# ## Task 2.  MLP prediction for  all the XOR examples.

# #### Call function TrainModel to Train on the input examples with all the generated layers and weights 

# ### Note :If Epoch range exceeds and code is terminated so it needs to be run again

# In[202]:


hiddenweights,InputWeights = TrainModel(InputLayerWeights,hiddenlayerweights,ip_with_bias,output,100000,0.1,total_no_examples)
print("Best Optimized weights for hidden layer \n",hiddenweights)
print(" Best Optimized weights for input  layer \n",InputWeights)
# Now find the predicted output from trained model aftern finding the best optimised weights for first to second and second to final layer
final_output = PredictTrainFinalModel(ip_with_bias,InputWeights,hiddenweights,4)
# NOTE: Actual value is not in 0 and 1 .It is just for understanding and mapping it with the actual XOR abilities for 0 and 1 based output
print("XOR Results :" )
print("Inputs|predicted Output|Threshold Round Off Output Set(0.5)")
for i in range(len(inputs)):
    y=0
    y=final_output[i]
    if y>0.5:
       print(inputs[i],"|",y,"|",1)
    else:
       print(inputs[i],"|",y,"|",0) 
    y+=1 


# ## Task 3 : 200 Vectors  with 4 components with Output calculated as mentioned in the assignment sin(x1-x2+x3-x4)

# In[284]:


input = np.random.random((200,4))# generate 200*4 vectors ie 200 examples with 4 inputs per example
output1 = (np.sin(input[:,0:1]-input[:,1:2]+input[:,2:3]-input[:,3:4]))#select columns 1,2,3,4 and apply sin(x1-x2+x3-x4) operation
input_train ,output_train = input[:150],(output1[:150])#Select 150 examples for training on input and output
input_test,output_test = input[150:200],(output1[150:200])# Select 50 examples for test data for input and output
no_IP,no_OP,NO_examples = node_info(input_train,output_train) # return informationa about the nodes.

print("Inputs per node : ",no_IP,"Output per node :",no_OP,"Total Training Examples :",NO_examples )


# #### Call function GenerateWeights to create First layer,Second(hidden layer), in the model 

# In[285]:


ILWeights = GenerateWeights(no_IP,no_IP+1) # First Input Layer weights
HLweights = GenerateWeights(no_IP+1,no_OP)# Second hidden Layer weights
ip_bias= bias(input_train,NO_examples)# add bias to the input
test_input_bias=bias(input_test,50)# add bias to the input test data aswell
print("First Input Layer weights \n",ILWeights,"\n Second hidden Layer weights: \n",HLweights)


# ### MLP trained on 4 inputs, 5 hidden units and 1 output on 150 examples
# #### Note :If Epoch range exceeds and code is terminated so it needs to be run again

# In[286]:


print("*****Model Training for input Sin Examples (150)*** \n")
# calll training model as created earlier in task 1 and return best optimised weights.
Hweights,IWeights = TrainModel(ILWeights,HLweights,ip_bias,output_train,1000,1.0,NO_examples)
# Predict output on trained data and store in final_output
final_output = PredictTrainFinalModel(ip_bias,IWeights,Hweights,150)
print("Predict final y_train",final_output)


# ## Predict Test output from Created Model by calling PredictTrainFinalModel

# In[287]:


print("Predict Test output from Created Model \n")
test_predict = PredictTrainFinalModel(test_input_bias,IWeights,Hweights,50)
print("test_predict \ n",test_predict)


# ## Task 4 Find error on Test output and predicted Test output using Root mean squared error

# In[288]:


from sklearn.metrics import mean_squared_error as RM
x= RM(output_test,test_predict)
print("Test Error percentage ",x*100,"%")


# ## Task 4 Find error on Train output and predicted Predicted output using Root mean squared error

# In[289]:


trainerror=RM(output_train,final_output)
print("Train Error percentage",trainerror*100,"%")

