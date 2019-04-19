import numpy as np
import part1_data
from copy import copy
import pudb


def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.001
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.001
        
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def softmax(Z):
    exps = np.exp(Z)
    return exps / np.sum(exps, axis =0)
 

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(dA, Z):
    z = np.array(Z, copy= True)
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z)))
    dZ = dA * s.T * (1 - s.T)
    # pu.db
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    elif activation is "softmax":
        activation_func = softmax
    else:
        raise Exception('Non-supported activation function')
        
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory


def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y.T, np.log(Y_hat + 1e-6)) + np.dot(1 - Y.T, np.log(1 - Y_hat + 1e-6)))
    return np.squeeze(cost)

def convert_prob_into_class_2(Y):
    Y_hat_ = np.zeros(Y.T.shape)[0]
    Y_hat_[np.argmax(Y)] = 1
    return np.array(Y_hat_)
    # pu.db

def get_accuracy_value_2(Y_hat, Y):
    Y_hat_ = convert_prob_into_class_2(Y_hat)
    # pu.db
    return (Y_hat_ == np.reshape(Y, (1, 2))).all(axis = 0).mean()

def convert_prob_into_class(Y):
    Y_hat_ = []
    for i in range(len(Y)):
        if Y[i] < 0.5:
            Y_hat_.append(0.0)
        else:
            Y_hat_.append(1.0)
    return np.array(Y_hat_)
    # pu.db

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    # pu.db
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    elif activation is "softmax":
        backward_activation_func = softmax_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
   
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat + 1e-6));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

def train(train_batch, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        print(i)
        accuracy_history = []
        for each in train_batch:
            X = each[0][0]
            Y = np.reshape(np.array(each[1]), (1,1))
            Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
            cost = get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
            
            grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
    return params_values, cost_history, accuracy_history

def train_2(train_batch, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    accuracy_history = []
    cost_history_all = []

    for i in range(epochs):
        print(i)
        cost_history = []
        accuracy_history = []
        for each in train_batch:
            X = each[0][0]
            # pu.db
            Y = np.reshape(np.array(each[1]), (2,1))
            Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
            cost = get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = get_accuracy_value_2(Y_hat, Y)
            accuracy_history.append(accuracy)
            
            grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        # pu.db
        cost_history_all.append(sum(cost_history) / len(train_batch))
        
    return params_values, cost_history_all, accuracy_history

def test_2(batch, nn_architecture, params_values):
    # params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    accuracy_history = []
    for each in batch:
        X = each[0][0]
        Y = np.reshape(np.array(each[1]), (2,1))
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value_2(Y_hat, Y)
        accuracy_history.append(accuracy)
        
    return cost_history, accuracy_history


def test(batch, nn_architecture, params_values):
    # params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    accuracy_history = []
    for each in batch:
        X = each[0][0]
        Y = np.reshape(np.array(each[1]), (1,1))
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
    return cost_history, accuracy_history


if __name__ == "__main__":
    # global nn_architecture
    train_batch, test_batch = prep_data()
    pv, ch, ah = train(train_batch, nn_architecture, 10, 0.1)
    ch_test, ah_test = test(test_batch, nn_architecture, pv)
    pu.db
    # infer(train_batch)
    # pu.db
