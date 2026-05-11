import numpy as np
import pandas as pd
from sklearn import preprocessing

MLNN_Version = '4.0'
print(f' Version of MLNN is {MLNN_Version}\n',
      'class NN -- without external scaler')

class NN:
    """Creates a two layer neuron network: the hidden layer and the output layer.
       The output layer may be multiple.
       The activation functions of the layers may be set.

       X and T are numpy column vectors or set of column vectors as numpy matrices.
       If the data is pandas dataframe, it should be turned to numpy:
          X=np.array(DataFrame[[x0,x1,...]])
          T=np.array(DataFrame[[t0,...]])
       The same for confidences.

       Attributes:
          layers: A list of layers. [2,3,1] -- 2 inputs, 3 neurons in hidden layer, 1 neuron in output layer
          hidden_activation: Activation function of the hidden layer. 'linear' (by default), 'sigmoid', 'prelu'.
          output_activation: Activation function of the output layer. 'linear' (by default) for regression, 'softmax' for classification.
          Wh, bh, Wo, bo: Matrices of weights.
          JWh, Jbh, JWo, Jbo: Matrices of weight Jacobian."""

    def __init__(self, layers, hidden_activation='linear', output_activation='linear'):
        """Return a new MLP object with the specified parameters.
           layers: A list of layers. [2,3,1] -- 2 inputs, 3 neurons in hidden layer, 1 neuron in output layer
           hidden_activation: Activation function of the hidden layer. 'linear' (by default), 'sigmoid', 'prelu'.
           output_activation: Activation function of the output layer. 'linear' (by default) for regression, 'softmax' for classification."""
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.imported = False  # before weights import

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1. / (1. + np.exp(-x))

    def sigmoid_deriv(self, x):
        """Sigmoid derivative function."""
        return x * (1 - x)

    def PReLU(self, x):
        """Parametric ReLU activation function.
           Parameter = 0.001"""
        np.copyto(x, np.maximum(0.001 * x, x))
        return x

    def PReLU_deriv(self, x):
        """Parametric ReLU derivative function:
           Matrix with ones for positives and 0.001s for negatives."""
        y = np.ones_like(x)
        y[x < 0] = 0.001
        return y

    def softmax(self, x):
        """Softmax activation function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def hidden_activations(self, X):
        """Compute the hidden activations H.
           self.hidden_activation may be 'linear' (default), 'sigmoid' or 'prelu'."""
        Zh = (X @ self.Wh) + self.bh
        if self.hidden_activation == 'sigmoid':
            return self.sigmoid(Zh)
        elif self.hidden_activation == 'prelu':
            return self.PReLU(Zh)
        return Zh  # default (linear) activation

    def output_activations(self, H):
        """Compute the output Y.
           self.output_activation may be 'linear' (default) or 'softmax'."""
        Zo = (H @ self.Wo) + self.bo
        if self.output_activation == 'softmax':
            return self.softmax(Zo)
        if self.output_activation == 'linear':
            return Zo
        return Zo  # default

    def __run(self, X):
        """Runs the input X vectors of input layer, returns the Y vectors of output layer."""
        return self.output_activations(self.hidden_activations(X))

    def loss(self, Y, T, confidences=None):
        """Loss function.
           Regression: MSE (output_activation -- 'linear').
           Classification: cross-entropy (output_activation -- 'softmax')."""
        SE = (Y - T) ** 2  # MSE
        abs_err = np.abs(Y - T)  # ABS
        confidences = np.ones_like(T) if confidences is None else confidences
        confidences = np.minimum(confidences, 1e4)  # avoid confidences=inf (variance=0)
        weighted_SE = SE * confidences
        MSE = weighted_SE.sum() / Y.shape[0]
        if self.output_activation == 'linear':
            return MSE
        if self.output_activation == 'softmax':
            return - (T * np.log(Y)).sum()  # softmax layer with corresponding cross-entropy loss function
        return MSE  # default

    def error_output(self, Y, T):
        """Error function at the output"""
        return (Y - T) * self.confidences  # same for softmax and MSE

    def gradient_weight_out(self, H, Eo):
        """Gradients for the weight parameters at the output layer"""
        return H.T @ Eo

    def gradient_bias_out(self, Eo):
        """Gradients for the bias parameters at the output layer"""
        return np.sum(Eo, axis=0, keepdims=True)

    def error_hidden(self, H, Eo):
        """Error at the hidden layer.
        H * (1-H) * (E . Wo^T) for sigmoid
        [1 for pos, 0.001 for neg] * (E . Wo^T) for PReLU
        (E . Wo^T) for linear (default)."""
        if self.hidden_activation == 'sigmoid':
            return np.multiply(self.sigmoid_deriv(H), (Eo @ self.Wo.T))
        if self.hidden_activation == 'prelu':
            return np.multiply(self.PReLU_deriv(H), (Eo @ self.Wo.T))
        if self.hidden_activation == 'linear':
            return (Eo @ self.Wo.T)
        return (Eo @ self.Wo.T)  # default

    def gradient_weight_hidden(self, X, Eh):
        """Gradient for the weight parameters at the hidden layer"""
        return X.T @ Eh

    def gradient_bias_hidden(self, Eh):
        """Gradient for the bias parameters at the output layer"""
        return np.sum(Eh, axis=0, keepdims=True)

    def generate_weights(self, init_var=0.1):
        # Initialize hidden layer parameters
        bh = np.random.randn(1, self.layers[1]) * init_var
        Wh = np.random.randn(self.layers[0], self.layers[1]) * init_var
        # Initialize output layer parameters
        bo = np.random.randn(1, self.layers[2]) * init_var
        Wo = np.random.randn(self.layers[1], self.layers[2]) * init_var
        return Wh, bh, Wo, bo

    def get_gradients(self, X, T):
        """Update the network parameters over 1 iteration."""
        # Compute the output of the network
        H = self.hidden_activations(X)
        Y = self.output_activations(H)
        # Compute the gradients of the output layer
        Eo = self.error_output(Y, T)
        self.JWo = self.gradient_weight_out(H, Eo)
        self.Jbo = self.gradient_bias_out(Eo)
        # Compute the gradients of the hidden layer
        Eh = self.error_hidden(H, Eo)
        self.JWh = self.gradient_weight_hidden(X, Eh)
        self.Jbh = self.gradient_bias_hidden(Eh)

    def update_momentum(self, X, T, Ms, lr_decay, momentum_term):
        self.get_gradients(X, T)
        Js = [self.JWh, self.Jbh, self.JWo, self.Jbo]
        return [momentum_term * M - lr_decay * J
                for M, J in zip(Ms, Js)]

    def update_weights(self, Ms):
        Ws = [self.Wh, self.bh, self.Wo, self.bo]
        return [P + M for P, M in zip(Ws, Ms)]

    def fit(self, X, T, epochs, confidences=None, X_valid=[], T_valid=[], learning_rate=0.01, learning_rate_decay=0, momentum_term=0.9,
            init_var=0.1, repeat=False):
        """Run backpropagation:
              1. Initilizes weights matrices (if repeat is False)
              2. Creates list of losses and calculates initial loss by rinning self.loss() for train data and validation data (if present)
              3. Creates lists of weight matrices and puts initial matrices
              4. Starts epoch iterations for weights and loss update
                a. Calculates learning rate decay:
                   lr_decay = learning_rate / (1 + learning_rate_decay * epoch)
                   learning_rate_decay = 0 in case of constant learning rate
                b. Runs update_momentum() function, which calls get_gradients() function.
                   get_gradients() calculates H, Y, Eo, Eh and returns weights' Jacobians: self.JWh, self.Jbh, self.JWo, self.Jbo
                   update_momentum() returns list of momentums and replaces the previous ones
                c. Runs update_weights() which returns new weight matrices and replaces the previous ones
                d. Calculates loss and addes to the list of losses for train data and validation data (if present)
                e. New weight matrices are added to their lists
              5. Lists of weight matrices are turned to .self numpy arrays for better slicing option

           epochs: int number of epochs
           X and T are numpy column vectors or set of column vectors as numpy matrices.
           If the data is pandas dataframe, it should be turned to numpy:
              X=np.array(DataFrame[[x0,x1,...]])
              T=np.array(DataFrame[[t0,...]])
           confidences: 1/variances. None by default. If present, the function calculates the confidence interval of the loss.
           learning_rate: learning rate, 0.01 by default
           learning_rate_decay: learning rate decay, 0 by default, integer or float
           momentum_term: momentum term, 0.9 by default, 0 for simple gradiend descent
           init_var: initial variance of generated weights, multiplies np.random.randn(), 0.1 by default
           repeat: False -- new weights are generated,
                   True -- old weights are used from previous fit() or import_weights()

           The results are:
           self.loss_list -- list of floats
           self.valid_loss_list -- list of floats (if validation data is present)
           """
        # Calculate scale data: mean and std
        scaler_X = preprocessing.StandardScaler().fit(X)
        scaler_T = preprocessing.StandardScaler().fit(T)
        self.scaler_X, self.scaler_T = scaler_X, scaler_T
        self.mean_X, self.std_X = scaler_X.mean_, scaler_X.scale_
        self.mean_T, self.std_T = scaler_T.mean_, scaler_T.scale_
        X = scaler_X.transform(X)
        T = scaler_T.transform(T)
        if len(X_valid) > 0:
            X_valid = scaler_X.transform(X_valid)
            T_valid = scaler_T.transform(T_valid)

        if not repeat:
            self.Wh, self.bh, self.Wo, self.bo = self.generate_weights(init_var)
        if self.imported:  # in case of weights import
            self.Wh = np.diag(self.std_X) @ self.Wh_export
            self.bh = self.bh_export + self.mean_X @ self.Wh_export
            self.Wo = self.Wo_export @ np.diag(1/self.std_T)
            self.bo = (self.bo_export - self.mean_T) @ np.diag(1/self.std_T)

        Ms = [np.zeros_like(M) for M in [self.Wh, self.bh, self.Wo, self.bo]]  # Momentums initialization

        # Run backpropagation
        confidences = np.ones_like(T) if confidences is None else np.array(confidences,dtype=np.float64)
        self.confidences = np.minimum(confidences, 1e4)  # avoid confidences=inf (variance=0)

        self.loss_list = [self.loss(self.__run(X), T, confidences)]
        self.valid_loss_list = [self.loss(self.__run(X_valid), T_valid)] if len(X_valid) > 0 else []

        for i in range(epochs):
            # Compute learning rate decay
            lr_decay = learning_rate / (1 + learning_rate_decay * i)

            # Update the momentums and weights
            Ms = self.update_momentum(X, T, Ms, lr_decay, momentum_term)
            self.Wh, self.bh, self.Wo, self.bo = self.update_weights(Ms)

            # Append loss
            self.loss_list.append(self.loss(self.__run(X), T, confidences))
            if len(X_valid) > 0:
                self.valid_loss_list.append(self.loss(self.__run(X_valid), T_valid))

        # Calculate export weights, considering scale
        self.Wh_export = np.diag(1/self.std_X) @ self.Wh
        self.bh_export = self.bh - self.mean_X @ (np.diag(1/self.std_X) @ self.Wh)
        self.Wo_export = self.Wo @ np.diag(self.std_T)
        self.bo_export = self.bo @ np.diag(self.std_T) + self.mean_T

    def export_weights(self):
        """
        return [self.Wh.tolist(), self.bh.tolist(), self.Wo.tolist(), self.bo.tolist()]
        Arranges weights without word "array", in a way that makes possible copy/paste and import as self.import_weights().
        """
        return [self.Wh_export.tolist(), self.bh_export.tolist(), self.Wo_export.tolist(), self.bo_export.tolist()]

    def export_weights_as_numpy(self):
        return [self.Wh_export, self.bh_export, self.Wo_export, self.bo_export]

    def export_weights_as_pandas(self):
        a=[self.Wh_export.tolist(), self.bh_export.tolist(), self.Wo_export.tolist(), self.bo_export.tolist()]
        b=pd.DataFrame()
        for i in a:
            b=pd.concat([b,pd.DataFrame(i)], axis=0)
        return b

    def print_weights(self):
        print('Layers (input, hidden, output): ', self.layers, self.hidden_activation, self.output_activation)
        print('Hidden layer weights: ', self.Wh_export.tolist())
        print('Hidden layer biases: ', self.bh_export.tolist())
        print('Outlet layer weights: ', self.Wo_export.tolist())
        print('Outlet layer biases: ', self.bo_export.tolist())

    def import_weights(self, weights):
        """
        Puts values to Wh, bh, Wo and bo from the list of lists as it is from self.export_weights().
        Flags that weights were imported, so Wh, bh, Wo and bo will be calculated from them in fit().
        """
        self.Wh_export, self.bh_export, self.Wo_export, self.bo_export = np.array(weights[0]), np.array(weights[1]), np.array(weights[2]), np.array(weights[3])
        self.imported = True

    def hidden_activations_export(self, X):
        """Compute the hidden activations H.
           self.hidden_activation may be 'linear' (default), 'sigmoid' or 'prelu'."""
        Zh = (X @ self.Wh_export) + self.bh_export
        if self.hidden_activation == 'sigmoid':
            return self.sigmoid(Zh)
        elif self.hidden_activation == 'prelu':
            return self.PReLU(Zh)
        return Zh  # default (linear) activation

    def output_activations_export(self, H):
        """Compute the output Y.
           self.output_activation may be 'linear' (default) or 'softmax'."""
        Zo = (H @ self.Wo_export) + self.bo_export
        if self.output_activation == 'softmax':
            return self.softmax(Zo)
        if self.output_activation == 'linear':
            return Zo
        return Zo  # default

    def predict(self, X):
        """Calculates prediction."""
        X = np.array(X).astype(float)
        return self.output_activations_export(self.hidden_activations_export(X))



class VectorBackProp:
    """Creates a two layer neuron network: the hidden layer and the output layer.
       The output layer may be multiple.
       The activation functions of the layers may be set.

       X and T are numpy column vectors or set of column vectors as numpy matrices.
       If the data is pandas dataframe, it should be turned to numpy:
          X=np.array(DataFrame[[x0,x1,...]])
          T=np.array(DataFrame[[t0,...]])
       The same for confidences.

       Attributes:
          layers: A list of layers. [2,3,1] -- 2 inputs, 3 neurons in hidden layer, 1 neuron in output layer
          hidden_activation: Activation function of the hidden layer. 'linear' (by default), 'sigmoid', 'prelu'.
          output_activation: Activation function of the output layer. 'linear' (by default) for regression, 'softmax' for classification.
          Wh, bh, Wo, bo: Matrices of weights.
          JWh, Jbh, JWo, Jbo: Matrices of weight Jacobian.
          Wh_history, bh_history, Wo_history, bo_history: 3D numpy arrays with histories of weights."""

    def __init__(self, layers, hidden_activation='linear', output_activation='linear'):
        """Return a new MLP object with the specified parameters.
           layers: A list of layers. [2,3,1] -- 2 inputs, 3 neurons in hidden layer, 1 neuron in output layer
           hidden_activation: Activation function of the hidden layer. 'linear' (by default), 'sigmoid', 'prelu'.
           output_activation: Activation function of the output layer. 'linear' (by default) for regression, 'softmax' for classification."""
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1. / (1. + np.exp(-x))

    def sigmoid_deriv(self, x):
        """Sigmoid derivative function."""
        return x * (1 - x)

    def PReLU(self, x):
        """Parametric ReLU activation function.
           Parameter = 0.001"""
        np.copyto(x, np.maximum(0.001 * x, x))
        return x

    def PReLU_deriv(self, x):
        """Parametric ReLU derivative function:
           Matrix with ones for positives and 0.001s for negatives."""
        y = np.ones_like(x)
        y[x < 0] = 0.001
        return y

    def softmax(self, x):
        """Softmax activation function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def hidden_activations(self, X):
        """Compute the hidden activations H.
           self.hidden_activation may be 'linear' (default), 'sigmoid' or 'prelu'."""
        Zh = (X @ self.Wh) + self.bh
        if self.hidden_activation == 'sigmoid':
            return self.sigmoid(Zh)
        elif self.hidden_activation == 'prelu':
            return self.PReLU(Zh)
        return Zh  # default (linear) activation

    def output_activations(self, H):
        """Compute the output Y.
           self.output_activation may be 'linear' (default) or 'softmax'."""
        Zo = (H @ self.Wo) + self.bo
        if self.output_activation == 'softmax':
            return self.softmax(Zo)
        if self.output_activation == 'linear':
            return Zo
        return Zo  # default

    def run(self, X):
        """Runs the input X vectors of input layer, returns the Y vectors of output layer."""
        return self.output_activations(self.hidden_activations(X))

    def nn_predict(self, X):
        """Neural network prediction function that only returns
        1 or 0 depending on the predicted class"""
        return np.around(self.run(X))

    def loss(self, Y, T, confidences=None):
        """Loss function.
           Regression: MSE (output_activation -- 'linear').
           Classification: cross-entropy (output_activation -- 'softmax')."""
        SE = (Y - T) ** 2  # MSE
        abs_err = np.abs(Y - T)  # ABS
        confidences = np.ones_like(T) if confidences is None else confidences
        confidences = np.minimum(confidences, 1e4)  # avoid confidences=inf (variance=0)
        weighted_SE = SE * confidences
        MSE = weighted_SE.sum() / Y.shape[0]
        
        if self.output_activation == 'linear':
            return MSE
        if self.output_activation == 'softmax':
            return - (T * np.log(Y)).sum()  # softmax layer with corresponding cross-entropy loss function
        
        return MSE  # default

    def error_output(self, Y, T):
        """Error function at the output"""
        return (Y - T) * self.confidences  # same for softmax and MSE

    def gradient_weight_out(self, H, Eo):
        """Gradients for the weight parameters at the output layer"""
        return H.T @ Eo

    def gradient_bias_out(self, Eo):
        """Gradients for the bias parameters at the output layer"""
        return np.sum(Eo, axis=0, keepdims=True)

    def error_hidden(self, H, Eo):
        """Error at the hidden layer.
        H * (1-H) * (E . Wo^T) for sigmoid
        [1 for pos, 0.001 for neg] * (E . Wo^T) for PReLU
        (E . Wo^T) for linear (default)."""
        if self.hidden_activation == 'sigmoid':
            return np.multiply(self.sigmoid_deriv(H), (Eo @ self.Wo.T))
        if self.hidden_activation == 'prelu':
            return np.multiply(self.PReLU_deriv(H), (Eo @ self.Wo.T))
        if self.hidden_activation == 'linear':
            return (Eo @ self.Wo.T)
        return (Eo @ self.Wo.T)  # default

    def gradient_weight_hidden(self, X, Eh):
        """Gradient for the weight parameters at the hidden layer"""
        return X.T @ Eh

    def gradient_bias_hidden(self, Eh):
        """Gradient for the bias parameters at the output layer"""
        return np.sum(Eh, axis=0, keepdims=True)

    def generate_weights(self, init_var=0.1):
        # Initialize hidden layer parameters
        bh = np.random.randn(1, self.layers[1]) * init_var
        Wh = np.random.randn(self.layers[0], self.layers[1]) * init_var
        # Initialize output layer parameters
        bo = np.random.randn(1, self.layers[2]) * init_var
        Wo = np.random.randn(self.layers[1], self.layers[2]) * init_var
        return Wh, bh, Wo, bo

    def get_gradients(self, X, T):
        """Update the network parameters over 1 iteration."""
        # Compute the output of the network
        # Compute the activations of the layers
        H = self.hidden_activations(X)
        Y = self.output_activations(H)
        # Compute the gradients of the output layer
        Eo = self.error_output(Y, T)
        self.JWo = self.gradient_weight_out(H, Eo)
        self.Jbo = self.gradient_bias_out(Eo)
        # Compute the gradients of the hidden layer
        Eh = self.error_hidden(H, Eo)
        self.JWh = self.gradient_weight_hidden(X, Eh)
        self.Jbh = self.gradient_bias_hidden(Eh)

    def update_momentum(self, X, T, Ms, lr_decay, momentum_term):
        """Update the momentum term."""
        # list_of_weights = [Wh, bh, Wo, bo]
        self.get_gradients(X, T)
        Js = [self.JWh, self.Jbh, self.JWo, self.Jbo]
        return [momentum_term * M - lr_decay * J
                for M, J in zip(Ms, Js)]

    def update_weights(self, Ms):
        """Update the weights."""
        Ws = [self.Wh, self.bh, self.Wo, self.bo]
        # Ms = [MWh, Mbh, MWo, Mbo]
        return [P + M for P, M in zip(Ws, Ms)]

    def fit(self, X, T, epochs, confidences=None, X_valid=[], T_valid=[], learning_rate=0.01, learning_rate_decay=0, momentum_term=0.9,
            init_var=0.1, repeat=False):
        """Run backpropagation:
              1. Initilizes weights matrices (if repeat is False)
              2. Creates list of losses and calculates initial loss by rinning self.loss() for train data and validation data (if present)
              3. Creates lists of weight matrices and puts initial matrices
              4. Starts epoch iterations for weights and loss update
                a. Calculates learning rate decay:
                   lr_decay = learning_rate / (1 + learning_rate_decay * epoch)
                   learning_rate_decay = 0 in case of constant learning rate
                b. Runs update_momentum() function, which calls get_gradients() function.
                   get_gradients() calculates H, Y, Eo, Eh and returns weights' Jacobians: self.JWh, self.Jbh, self.JWo, self.Jbo
                   update_momentum() returns list of momentums and replaces the previous ones
                c. Runs update_weights() which returns new weight matrices and replaces the previous ones
                d. Calculates loss and addes to the list of losses for train data and validation data (if present)
                e. New weight matrices are added to their lists
              5. Lists of weight matrices are turned to .self numpy arrays for better slicing option

           epochs: int number of epochs
           X and T are numpy column vectors or set of column vectors as numpy matrices.
           If the data is pandas dataframe, it should be turned to numpy:
              X=np.array(DataFrame[[x0,x1,...]])
              T=np.array(DataFrame[[t0,...]])
           confidences: 1/variances. None by default. If present, the function calculates the confidence interval of the loss. 
           learning_rate: learning rate, 0.01 by default
           learning_rate_decay: learning rate decay, 0 by default, integer or float
           momentum_term: momentum term, 0.9 by default, 0 for simple gradiend descent
           init_var: initial variance of generated weights, multiplies np.random.randn(), 0.1 by default
           repeat: False -- new weights are generated,
                   True -- old weights are used from previous fit() or import_weights()

           The results are:
           self.loss_list -- list of floats
           self.valid_loss_list -- list of floats (if validation data is present)
           self.Wh_history -- 3D numpy array of weights [epochs, starting neuron, target neuron]
           self.bh_history
           self.Wo_history
           self.bo_history

           self.JWh_history -- 3D numpy array of weights' Jacobians
           self.Jbh_history
           self.JWo_history
           self.Jbo_history
           """
        # Run backpropagation
        if not repeat:
            self.Wh, self.bh, self.Wo, self.bo = self.generate_weights(init_var)
        Ms = [np.zeros_like(M) for M in [self.Wh, self.bh, self.Wo, self.bo]]  # Momentums initialization
        
        confidences = np.ones_like(T) if confidences is None else np.array(confidences,dtype=np.float64)
        self.confidences = np.minimum(confidences, 1e4)  # avoid confidences=inf (variance=0)
        
        self.loss_list = [self.loss(self.run(X), T, confidences)]
        self.valid_loss_list = [self.loss(self.run(X_valid), T_valid)] if len(X_valid) > 0 else []

        Wh_hist, bh_hist, Wo_hist, bo_hist = [self.Wh], [self.bh], [self.Wo], [self.bo]
        JWh_hist, Jbh_hist, JWo_hist, Jbo_hist = [], [], [], []

        for i in range(epochs):
            # Compute learning rate decay
            lr_decay = learning_rate / (1 + learning_rate_decay * i)

            # Update the momentums and weights
            Ms = self.update_momentum(X, T, Ms, lr_decay, momentum_term)
            self.Wh, self.bh, self.Wo, self.bo = self.update_weights(Ms)

            # Append loss
            self.loss_list.append(self.loss(self.run(X), T, confidences))
            if len(X_valid) > 0:
                self.valid_loss_list.append(self.loss(self.run(X_valid), T_valid))

            # Record weight histories
            Wh_hist.append(self.Wh)
            bh_hist.append(self.bh)
            Wo_hist.append(self.Wo)
            bo_hist.append(self.bo)

            JWh_hist.append(self.JWh)
            Jbh_hist.append(self.Jbh)
            JWo_hist.append(self.JWo)
            Jbo_hist.append(self.Jbo)

        # Store weight histories
        self.Wh_history = np.array(Wh_hist)
        self.bh_history = np.array(bh_hist)
        self.Wo_history = np.array(Wo_hist)
        self.bo_history = np.array(bo_hist)

        self.JWh_history = np.array(JWh_hist)
        self.Jbh_history = np.array(Jbh_hist)
        self.JWo_history = np.array(JWo_hist)
        self.Jbo_history = np.array(Jbo_hist)

    def export_weights(self):
        """
        return [self.Wh.tolist(), self.bh.tolist(), self.Wo.tolist(), self.bo.tolist()]
        Arranges weights without word "array", in a way that makes possible copy/paste and import as self.import_weights().
        """
        return [self.Wh.tolist(), self.bh.tolist(), self.Wo.tolist(), self.bo.tolist()]

    def export_weights_as_numpy(self):
        """
        return [self.Wh, self.bh, self.Wo, self.bo]"""
        return [self.Wh, self.bh, self.Wo, self.bo]
    
    def export_weights_as_pandas(self):
        a=[self.Wh.tolist(), self.bh.tolist(), self.Wo.tolist(), self.bo.tolist()]
        b=pd.DataFrame()
        for i in a:
            b=pd.concat([b,pd.DataFrame(i)], axis=0)
        return b

    def print_weights(self):
        """
        Prints the layers and weights.
        """
        print('Layers (input, hidden, output): ', self.layers, self.hidden_activation, self.output_activation)
        print('Hidden layer weights: ', self.Wh.tolist())
        print('Hidden layer biases: ', self.bh.tolist())
        print('Outlet layer weights: ', self.Wo.tolist())
        print('Outlet layer biases: ', self.bo.tolist())

    def import_weights(self, weights):
        """
        Puts values to Wh, bh, Wo and bo from the list of lists as it is from self.export_weights().
        """
        self.Wh, self.bh, self.Wo, self.bo = np.array(weights[0]), np.array(weights[1]), np.array(weights[2]), np.array(weights[3])
