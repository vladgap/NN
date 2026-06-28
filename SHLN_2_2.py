import numpy as np
import pandas as pd
from sklearn import preprocessing
from plotly.subplots import make_subplots
import plotly.graph_objects as go

SHLN_Version = '2.2'
print(f' SHLN -- single-hidden-layer network. Version: {SHLN_Version}\n',
      'together with NN2to1, fixes')

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
           hidden_activation: Activation function of the hidden layer.
                              String: 'linear' (by default), 'sigmoid', 'prelu' -- same for all neurons.
                              List:   e.g. ['sigmoid', 'linear', 'prelu'] -- one activation per neuron,
                                      length must equal layers[1].
           output_activation: Activation function of the output layer. 'linear' (by default) for regression, 'softmax' for classification."""
        self.layers = layers
        self.output_activation = output_activation
        self.imported = False  # before weights import
        self.confidences = None
        # Validate and store hidden_activation
        if isinstance(hidden_activation, list):
            assert len(hidden_activation) == layers[1], \
                f"hidden_activation list length ({len(hidden_activation)}) must equal number of hidden neurons ({layers[1]})"
        self.hidden_activation = hidden_activation

    def sigmoid(self, x):
        return np.where(x >= 0,
                        1. / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def sigmoid_deriv(self, x):
        """Sigmoid derivative function."""
        return x * (1 - x)

    def PReLU(self, x):
        """Parametric ReLU activation function.
           Parameter = 0.001"""
        return np.maximum(0.001 * x, x)

    def PReLU_deriv(self, x):
        """Parametric ReLU derivative function:
           Matrix with ones for positives and 0.001s for negatives."""
        y = np.ones_like(x)
        y[x < 0] = 0.001
        return y

    def softmax(self, x):
        """Softmax activation function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def _apply_activation(self, Zh, act):
        """Apply a single activation function (string) to matrix or vector Zh."""
        if act == 'sigmoid':
            return self.sigmoid(Zh)
        elif act == 'prelu':
            return self.PReLU(Zh)
        return Zh  # linear (default)

    def _apply_activation_deriv(self, H, act):
        """Apply derivative of a single activation function (string) to H."""
        if act == 'sigmoid':
            return self.sigmoid_deriv(H)
        elif act == 'prelu':
            return self.PReLU_deriv(H)
        return np.ones_like(H)  # linear derivative = 1

    def hidden_activations(self, X):
        """Compute the hidden activations H.
           self.hidden_activation may be:
             string: 'linear' (default), 'sigmoid', 'prelu' -- same for all neurons
             list:   one activation string per neuron, e.g. ['sigmoid', 'linear', 'prelu']"""
        Zh = (X @ self.Wh) + self.bh
        if isinstance(self.hidden_activation, list):
            H = np.zeros_like(Zh)
            for i, act in enumerate(self.hidden_activation):
                H[:, i] = self._apply_activation(Zh[:, i], act)
            return H
        return self._apply_activation(Zh, self.hidden_activation)

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
        if self.confidences is None:
            return (Y - T)
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
        (E . Wo^T) for linear (default).
        If hidden_activation is a list, applies per-neuron derivative."""
        back = Eo @ self.Wo.T
        if isinstance(self.hidden_activation, list):
            D = np.zeros_like(H)
            for i, act in enumerate(self.hidden_activation):
                D[:, i] = self._apply_activation_deriv(H[:, i], act)
            return np.multiply(D, back)
        return np.multiply(self._apply_activation_deriv(H, self.hidden_activation), back)

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

    def fit(self, X, T, epochs, confidences=None, X_valid=None, T_valid=None, learning_rate=0.01, learning_rate_decay=0, momentum_term=0.9,
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
        if X_valid is None: X_valid = []
        if T_valid is None: T_valid = []
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
        """Compute the hidden activations H using export weights (unscaled space).
           self.hidden_activation may be string or list (same logic as hidden_activations)."""
        Zh = (X @ self.Wh_export) + self.bh_export
        if isinstance(self.hidden_activation, list):
            H = np.zeros_like(Zh)
            for i, act in enumerate(self.hidden_activation):
                H[:, i] = self._apply_activation(Zh[:, i], act)
            return H
        return self._apply_activation(Zh, self.hidden_activation)

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


############################################################################################################
############################################################################################################


class NN2to1:
    def __init__(self, X, T, mesh, confidences=None, hidden_layers=1, hidden_activation='linear'):
        self.X=X
        self.T=T
        self.mesh=mesh
        self.confidences=confidences
        self.hidden_layers=hidden_layers
        self.hidden_activation=hidden_activation
        self.network=NN(layers=[2,hidden_layers,1], hidden_activation = hidden_activation)
        pd.options.plotting.backend = "plotly"

    def fit_model(self, epochs=1000, learning_rate=0.001, momentum_term=0.95):
        self.network.fit(self.X, self.T, epochs=epochs, confidences=self.confidences,
                         learning_rate=learning_rate, momentum_term=momentum_term)
        print('Initial loss =', self.network.loss_list[0])
        print('Final loss =', self.network.loss_list[-1])
        self.show()

    def import_weights(self,weights):
        self.network.import_weights(weights)

    def export_weights(self):
        print ('Hidden layers:', self.hidden_layers)
        print ('Hidden activation:', self.hidden_activation)
        print ('Loss:', self.network.loss_list[-1],'\n')
        return self.network.export_weights()

    def print_weights(self):
        self.network.print_weights()

    def show(self):
        self.predics=self.network.predict(self.X)
        self.errors = np.where(self.T[:,0] != 0,
                    (self.predics[:,0] - self.T[:,0]) / self.T[:,0] * 100,
                    np.nan)
        self.mesh_predics=self.network.predict(self.mesh)
        self.__plot()

    def __plot(self):
        has_loss = hasattr(self.network, 'loss_list')

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Model', 'Errors', ''),
            column_widths=[0.5, 0.5],
            row_heights=[0.5, 0.5],
            vertical_spacing=0.1,
            specs=[
                [{"type": "xy"},        {"type": "scene", "rowspan": 2}],
                [{"secondary_y": True}, None]
            ]
        )

        # --- Loss subplot [1,1] ---
        if has_loss:
            fig.add_trace(go.Scatter(y=self.network.loss_list, mode='lines',
                                     name='Loss', line_color='steelblue'), 1, 1)

        # --- Errors subplot [2,1] ---
        if self.confidences is None:
            error_marker_size = 4
        else:
            c = np.asarray(self.confidences).flatten()
            # error_marker_size = 4 if c.max() == c.min() else 1 + 6 * (c - c.min()) / (c.max() - c.min())
            error_marker_size = (4 + (c - c.min()) / (c.max() - c.min() + 1E-4) * (10-4)).astype(float) # confidence

        fig.add_trace(go.Scatter(x=self.T[:,0], y=self.predics[:,0], mode='markers',
                                 marker_size=error_marker_size, name='Predics', marker_color='black'), 2, 1)
        fig.add_trace(go.Scatter(x=self.T[:,0], y=self.T[:,0], mode='lines',
                                 line_color='red', line_width=0.2, showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=self.T[:,0], y=self.errors, mode='markers',
                                 marker_size=error_marker_size, name='Errors', marker_color='orange'), 2, 1, secondary_y=True)

        # --- 3D Model subplot [1,2] rowspan=2 ---
        fig.add_trace(go.Scatter3d(x=self.X[:,0], y=self.X[:,1], z=self.T[:,0],
                                   mode='markers', marker_color='blue', name='Data'), 1, 2)
        fig.add_trace(go.Scatter3d(x=self.mesh[:,0], y=self.mesh[:,1], z=self.mesh_predics[:,0],
                                   mode='markers', marker_color='green', marker_size=1, name='Mesh'), 1, 2)

        fig.update_layout(autosize=True, height=500, margin=dict(l=0, r=0, b=0, t=30))
        fig.update_scenes(xaxis_title='x', yaxis_title='y',
                          camera_eye=dict(x=-1, y=-1, z=1),
                          aspectratio=dict(x=1, y=1, z=1))
        fig.update_scenes(camera_projection_type="orthographic")
        self.fig = fig
        fig.show()
        self.fig = fig
        fig.show()
