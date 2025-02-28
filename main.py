import numpy as np
import os
cnt = 0

f = open("set.txt","r")
a = f.readline()
b = f.readline()
b = b.strip("\n")
b = int(b)
a = a.strip("\n")
cnt = int(a)
f.close()
np.random.seed(b)

W1 = np.random.randn(784, 16) * 0.01  
b1 = np.zeros((1, 16))

W2 = np.random.randn(16, 16) * 0.01  
b2 = np.zeros((1, 16))

W3 = np.random.randn(16, 10) * 0.01 
b3 = np.zeros((1, 10))

def relu(x):
    return np.maximum(0, x) 


import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28*28)  # Flatten if needed
X_test = X_test.reshape(-1, 28*28)  # Flatten if needed

print("MNIST dataset downloaded automatically!")


def relu(x):
    return np.maximum(0, x)  # ReLU activation

def relu_derivative(x):
    return (x > 0).astype(float)  # ReLU derivative

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability fix
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_pass(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def backward_pass(X, y_true, Z1, A1, Z2, A2, Z3, A3, learning_rate=0.01):
    # One-hot encode y_true
    global W1, b1, W2, b2, W3, b3
    y_true_one_hot = np.eye(10)[y_true]

    # Compute output layer error
    dZ3 = A3 - y_true_one_hot
    dW3 = np.dot(A2.T, dZ3) / X.shape[0]
    db3 = np.sum(dZ3, axis=0, keepdims=True) / X.shape[0]

    # Backpropagate to Hidden Layer 2
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

    # Backpropagate to Hidden Layer 1
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3


epochs = 1000
learning_rate = 0.02

#W1 = np.load(f"np/W1_{cnt}.npy")
#b1 = np.load(f"np/b1_{cnt}.npy")
#W2 = np.load(f"np/W2_{cnt}.npy")
#b2 = np.load(f"np/b2_{cnt}.npy")
#W3 = np.load(f"np/W3_{cnt}.npy")
#b3 = np.load(f"np/b3_{cnt}.npy")

print("CNT:" ,cnt)
n = open(f"record{b}.txt","w")
prt = ""
os.mkdir(f"np{b}")
for epoch in range(epochs):
    Z1, A1, Z2, A2, Z3, A3 = forward_pass(X_train)
    backward_pass(X_train, y_train, Z1, A1, Z2, A2, Z3, A3, learning_rate)
    if(epoch%5 == 0 ):
            print("CNT:",cnt)
            prt+= f"Epoch: {epoch}  Accuracy:"
            print("EPOCHOS =",epoch)
            _, _, _, _, _, test_predictions = forward_pass(X_test)
            y_pred = np.argmax(test_predictions, axis=1)
            
            accuracy = np.mean(y_pred == y_test)
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            prt+= f" {accuracy * 100:.2f}\n"
    if(epoch%20 == 0):
            cnt+=1
            np.save(f"np{b}/W1_{cnt}.npy", W1)
            np.save(f"np{b}/b1_{cnt}.npy", b1)
            np.save(f"np{b}/W2_{cnt}.npy", W2)
            np.save(f"np{b}/b2_{cnt}.npy", b2)
            np.save(f"np{b}/W3_{cnt}.npy", W3)
            np.save(f"np{b}/b3_{cnt}.npy", b3)

prt = f"-----------RECORD {b} STARTS-----------\n"+ f"Total Epochs: {epochs}, LEARNING RATE: {learning_rate} , FINAL ACCURACY: {accuracy*100} \n" + prt
n.write(prt)
n.close()

   # if epoch % 100 == 0:
   #     loss = -np.mean(np.sum(one_hot(y_train) * np.log(A3 + 1e-8), axis=1))
   #     print(f"Epoch {epoch}, Loss: {loss:.4f}")

f = open("set.txt","w")
b+=1
f.write(str(cnt))
f.write("\n"+str(b))
f.close()
_, _, _, _, _, test_predictions = forward_pass(X_test)
y_pred = np.argmax(test_predictions, axis=1)

accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

