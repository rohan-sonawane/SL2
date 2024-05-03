#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT NO 1

# Write a program to scheme a few activation functions that are used in neural networks

# sigmoid Function

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10)
def sigmoid(x):
        return 1/(1+np.exp(-x))
plt.plot(x, sigmoid(x))
plt.axis('tight')
plt.title('Activation Function :Sigmoid')
plt.show()


# Tanh Activation Function

# In[13]:


def tanh(x):
        return np.tanh(x)
x = np.linspace(-10, 10)
plt.plot(x, tanh(x))
plt.axis('tight')
plt.title('Activation Function :Tanh')
plt.show()


# RELU Activation Function

# In[15]:


def RELU(x):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)

    return x1
x = np.linspace(-10, 10)
plt.plot(x, RELU(x))
plt.axis('tight')
plt.title('Activation Function :RELU')
plt.show()


# Softmax Activation Function

# In[16]:


def softmax(x):
    ''' Compute softmax values for each sets of scores in x. '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)
x = np.linspace(-10, 10)
plt.plot(x, softmax(x))
plt.axis('tight')
plt.title('Activation Function :Softmax')
plt.show()


# Linear Activation Function

# In[17]:


def linear(x):
    ''' y = f(x) It returns the input as it is'''
    return x
x = np.linspace(-10, 10)
plt.plot(x, linear(x))
plt.axis('tight')
plt.title('Activation Function :Linear')
plt.show()


# Binary Step Activation Function

# In[18]:


def binaryStep(x):
    ''' It returns '0' is the input is less then zero otherwise it returns one '''
    return np.heaviside(x,1)
x = np.linspace(-10, 10)
plt.plot(x, binaryStep(x))
plt.axis('tight')
plt.title('Activation Function :binaryStep')
plt.show()


# # ASSIGNMENT NO 2

# Write a program to show back propagation network for XOR function with binary input 
# and output 

# In[45]:


import numpy as np
def sigmoid (x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])
epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

#Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))
print("Initial hidden weights:")
print(*hidden_weights)
print("Initial hidden biases:")
print(*hidden_bias)
print("Initial output weights:")
print(*output_weights)
print("Initial output biases:")
print(*output_bias)

#Training algorithm
for _ in range(epochs):
    #Forward Propagation
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    #Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    #Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr
print("\nFinal hidden weights:")
print(*hidden_weights)
print("Final hidden bias:")
print(*hidden_bias)
print("Final output weights:")
print(*output_weights)
print("Final output bias:")
print(*output_bias)
print("\nOutput from neural network after 10,000 epochs:")
print(*predicted_output)


# # ASSIGNMENT NO 3

# Write a program for producing back propagation feed forward network

# In[47]:


#Back propagation algorithm- code
# importing the library
import numpy as np

# creating the input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
print ("\n Input:")
print(X)

# creating the output array
y=np.array([[1],[1],[0]])
print ("\n Actual Output:")
print(y)

# defining the Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

# initializing the variables
epoch=5000 # number of training iterations
lr=0.1 # learning rate
inputlayer_neurons = X.shape[1] # number of features in data set
hiddenlayer_neurons = 3 # number of hidden layers neurons

output_neurons = 1 # number of neurons at output layer

# initializing weight and bias
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))#weight of hidden layer
bh=np.random.uniform(size=(1,hiddenlayer_neurons))#bise of hidden layer
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))#weight of output layer
bout=np.random.uniform(size=(1,output_neurons))#bise of output layer

# training the model
for i in range(epoch):

    #Forward Propagation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)
    
    #Backpropagation
    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    
print ("\n Output from the model:")
print (output)


# # ASSIGNMENT NO 4

# Write a program to demonstrate the perceptron learning law with its decision region
# using python. Give the output in graphical form
# 

# # 1

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def load_data():
    URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header = None)
    #print(data)
    
    # make the dataset linearly separable
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 0, 1)
    data = np.asmatrix(data, dtype = 'float64')
    return data
data = load_data()


# In[3]:


plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa')
plt.scatter(np.array(data[50:,0]), np.array(data[50:,2]), marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.show()


# In[79]:


def perceptron(data, num_iter):
    features = data[:, :-1]
    labels = data[:, -1]
    
    # set weights to zero
    w = np.zeros(shape=(1, features.shape[1]+1))
    
    misclassified_ = [] 
  
    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x,0,1)
            y = np.dot(w, x.transpose())
            target = 1.0 if (y > 0) else 0.0
            
            delta = (label.item(0,0) - target)
            
            if(delta): # misclassified
                misclassified += 1
                w += (delta * x)
        
        misclassified_.append(misclassified)
    return (w, misclassified_)
             
num_iter = 10
w, misclassified_ = perceptron(data, num_iter)


# In[80]:


epochs = np.arange(1, num_iter+1)
plt.plot(epochs, misclassified_)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.show()


# # 2

# In[85]:


import numpy as np
import matplotlib.pyplot as plt

# Generate random data points for two classes
np.random.seed(42)
class1_points = np.random.rand(20, 2) * 2 - 1
class2_points = np.random.rand(20, 2) * 2 + 1

# Create labels for the two classes
class1_labels = np.ones(20)
class2_labels = -np.ones(20)

# Combine data points and labels
X = np.vstack((class1_points, class2_points))
y = np.hstack((class1_labels, class2_labels))

# Initialize weights and bias
w = np.random.rand(2)
b = np.random.rand()

# Perceptron learning algorithm
learning_rate = 0.1
epochs = 100

for _ in range(epochs):
    for i in range(len(X)):
        if y[i] * (np.dot(w, X[i]) + b) <= 0:
            w += learning_rate * y[i] * X[i]
            b += learning_rate * y[i]

# Plot decision boundary
x_vals = np.linspace(-2, 2, 100)
y_vals = -(w[0] * x_vals + b) / w[1]

plt.scatter(class1_points[:, 0], class1_points[:, 1], label="Class 1", marker="o")
plt.scatter(class2_points[:, 0], class2_points[:, 1], label="Class 2", marker="x")
plt.plot(x_vals, y_vals, color="red", label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Learning Law")
plt.legend()
plt.grid(True)
plt.show()

