import numpy as np
import math


class RNNScratch:
    def __init__(self, input_features: int, hidden_units: int, num_output: int,
                 learning_rate: float, loss="mse", activation="tanh",
                 output_activation='sigmoid'):
        self.input_features = input_features
        self.hidden_units = hidden_units
        self.num_output = num_output
        self.loss = loss
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate

        # Activation functions
        self.hidden_act = {
            'linear': lambda x: x,
            'tanh': self.tanh,
            'sigmoid': self.sigmoid,
            'relu': self.relu
        }
        self.der_hidden_act = {
            'linear': lambda x: 1,
            'tanh': self.der_tanh,
            'sigmoid': self.der_sigmoid,
            'relu': self.der_relu
        }
        self.output_act = {
            'linear': lambda x: x,
            'sigmoid': self.sigmoid,
            'softmax': self.softmax
        }
        self.der_output_act = {
            'linear': lambda x: 1,
            'sigmoid': self.der_sigmoid,
            'softmax': self.der_softmax_ce
        }

        # Loss functions
        self.cal_loss = {
            'mse': self.mse,
            'ce': self.ce
        }
        self.der_los = {
            'mse': self.der_mse,
            'ce': self.der_ce
        }

        self.weights, self.bias = self.create_rnn()
        self.validate()

    def validate(self):
        if self.activation not in self.hidden_act:
            raise ValueError(f'Invalid activation function: {self.activation}')
        if self.output_activation not in self.output_act:
            raise ValueError(f'Invalid output activation: {self.output_activation}')
        if self.loss not in self.cal_loss:
            raise ValueError(f'Invalid loss function: {self.loss}')

    # Activation Functions
    def tanh(self, x):
        return np.tanh(x)

    def sigmoid(self, x):
        x = np.array(x, dtype=float)
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = np.array(x).flatten()
        x = x - np.max(x)
        exp_x = np.exp(x)
        return (exp_x / np.sum(exp_x)).reshape(-1, 1)

    # Derivative Functions
    def der_tanh(self, post_act):
        return 1 - (post_act ** 2)

    def der_sigmoid(self, post_act):
        return post_act * (1 - post_act)

    def der_relu(self, post_act):
        return (post_act > 0).astype(float)

    def der_softmax_ce(self, post_act, label):
        return post_act - label

    # Loss Functions
    def der_mse(self, predict, label):
        return (2 / self.num_output) * (predict - label)

    def der_ce(self, predict, label):
        eps = 1e-9
        return -label / (predict + eps)

    def mse(self, predicts, labels):
        self.loss_list = {'local': [], 'global': 0}

        for predict, label in zip(predicts, labels):
            predict_flat = np.array(predict).flatten()
            label_flat = np.array(label).flatten()

            loss = np.mean((predict_flat - label_flat) ** 2)
            self.loss_list['local'].append(loss)

        self.loss_list['global'] = np.mean(self.loss_list['local'])
        return self.loss_list

    def ce(self, predicts, labels):
        eps = 1e-12
        self.loss_list = {'local': [], 'global': 0}

        for predict, label in zip(predicts, labels):
            predict_flat = np.array(predict).flatten()
            label_flat = np.array(label).flatten()

            loss = -np.mean(label_flat * np.log(predict_flat + eps))
            self.loss_list['local'].append(loss)

        self.loss_list['global'] = np.mean(self.loss_list['local'])
        return self.loss_list

    # Weight Initialization
    def xavier_uniform(self, fan_in, fan_out):
        limit = math.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(low=-limit, high=limit, size=(fan_out, fan_in))

    def create_weights(self):
        return {
            'wx': self.xavier_uniform(self.input_features, self.hidden_units),
            'wh': self.xavier_uniform(self.hidden_units, self.hidden_units),
            'wy': self.xavier_uniform(self.hidden_units, self.num_output)
        }

    def create_bias(self):
        return {
            'bh': np.zeros((self.hidden_units, 1)),
            'by': np.zeros((self.num_output, 1))
        }

    def create_rnn(self):
        self.weights = self.create_weights()
        self.bias = self.create_bias()
        return self.weights, self.bias

    # Forward Pass
    def forward(self, X):
        X = np.array(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f'Input must be 2D array, got {X.ndim}D')

        self.input_data = X
        self.forward_result = {
            'pre_act': {'hidden_units': [], 'output': []},
            'post_act': {'hidden_units': [], 'output': []}
        }

        # Forward propagation through time (FPTT)
        for t, x in enumerate(X):
            x = x.reshape(-1, 1)

            # Calculate hidden state
            if t == 0:
                hidden = (self.weights['wx'] @ x) + self.bias['bh']
            else:
                h_prev = self.forward_result['post_act']['hidden_units'][-1]
                hidden = (self.weights['wx'] @ x) + (self.weights['wh'] @ h_prev) + self.bias['bh']

            # Store pre and post activation for hidden
            self.forward_result['pre_act']['hidden_units'].append(hidden)
            h_activated = self.hidden_act[self.activation](hidden)
            self.forward_result['post_act']['hidden_units'].append(h_activated)

            # Calculate output
            output = (self.weights['wy'] @ h_activated) + self.bias['by']
            self.forward_result['pre_act']['output'].append(output)

            # Apply output activation
            y_activated = self.output_act[self.output_activation](output)
            self.forward_result['post_act']['output'].append(y_activated)

        return self.forward_result['post_act']['output']

    def calculate_loss(self, predicts, labels):
        self.labels = np.array(labels, dtype=float)
        return self.cal_loss[self.loss](predicts, self.labels)

    # Backward Pass
    def backprop_find_gradient(self):
        n_timestep = len(self.forward_result['post_act']['output'])
        
        self.gradient['z'] = []
        self.gradient['hidden_state'] = []
        
        # Calculate output gradients
        for t in range(n_timestep):
            y_pred = self.forward_result['post_act']['output'][t].flatten()
            y_true = np.array(self.labels[t]).flatten()
            
            if self.loss == 'ce' and self.output_activation == 'softmax':
                grad_z = self.der_softmax_ce(y_pred, y_true)
            else:
                if callable(self.der_output_act[self.output_activation]):
                    der_act = self.der_output_act[self.output_activation](y_pred)
                else:
                    der_act = self.der_output_act[self.output_activation]
                
                der_loss = self.der_los[self.loss](y_pred, y_true)
                grad_z = der_act * der_loss
            
            self.gradient['z'].append(grad_z)
        
        # BPTT - Initialize with zeros for last timestep
        dh_next = np.zeros(self.hidden_units)
        
        # Backward through time
        for t in range(n_timestep - 1, -1, -1):
            h_t = self.forward_result['post_act']['hidden_units'][t].flatten()
            dh = self.weights['wy'].T @ self.gradient['z'][t]
            dh += self.weights['wh'].T @ dh_next
            dh = dh * self.der_hidden_act[self.activation](h_t)
            self.gradient['hidden_state'].insert(0, dh)
            
            dh_next = dh
            
    def backprop_find_gradient_weight(self):
        n_timestep = len(self.forward_result['post_act']['output'])

        self.gradient['wy'] = np.zeros_like(self.weights['wy'])
        self.gradient['wx'] = np.zeros_like(self.weights['wx'])
        self.gradient['wh'] = np.zeros_like(self.weights['wh'])

        for t in range(n_timestep):
            h_t = self.forward_result['post_act']['hidden_units'][t].flatten()
            grad_z_t = self.gradient['z'][t].reshape(-1, 1)
            self.gradient['wy'] += grad_z_t @ h_t.reshape(1, -1)

        for t in range(n_timestep):
            x_t = self.input_data[t].reshape(-1, 1)
            grad_h_t = self.gradient['hidden_state'][t].reshape(-1, 1)
            self.gradient['wx'] += grad_h_t @ x_t.T

        for t in range(1, n_timestep):
            h_prev = self.forward_result['post_act']['hidden_units'][t - 1].flatten()
            grad_h_t = self.gradient['hidden_state'][t].reshape(-1, 1)
            self.gradient['wh'] += grad_h_t @ h_prev.reshape(1, -1)

    def backprop_find_gradient_bias(self):
        n_timestep = len(self.forward_result['post_act']['output'])

        self.gradient['by'] = np.zeros_like(self.bias['by'])
        self.gradient['bh'] = np.zeros_like(self.bias['bh'])

        for t in range(n_timestep):
            self.gradient['by'] += self.gradient['z'][t].reshape(-1, 1)

        for t in range(n_timestep):
            self.gradient['bh'] += self.gradient['hidden_state'][t].reshape(-1, 1)

    def backprop(self):
        self.gradient = {}

        self.backprop_find_gradient()
        self.backprop_find_gradient_weight()
        self.backprop_find_gradient_bias()

    def clip_gradients(self, max_norm=5.0):
        # Clip weight gradients
        for key in ['wx', 'wy', 'wh']:
            grad_norm = np.linalg.norm(self.gradient[key])
            if grad_norm > max_norm:
                self.gradient[key] = self.gradient[key] * (max_norm / grad_norm)

        # Clip bias gradients
        for key in ['bh', 'by']:
            grad_norm = np.linalg.norm(self.gradient[key])
            if grad_norm > max_norm:
                self.gradient[key] = self.gradient[key] * (max_norm / grad_norm)

    def sgd(self, clip_gradient=True, max_grad_norm=5.0):
        # Clip gradients to prevent explosion
        if clip_gradient:
            self.clip_gradients(max_grad_norm)

        # Update weights
        self.weights['wx'] -= self.learning_rate * self.gradient['wx']
        self.weights['wy'] -= self.learning_rate * self.gradient['wy']
        self.weights['wh'] -= self.learning_rate * self.gradient['wh']

        # Update biases
        self.bias['by'] -= self.learning_rate * self.gradient['by']
        self.bias['bh'] -= self.learning_rate * self.gradient['bh']

    def optimize(self, clip_gradient=True):
        return self.sgd(clip_gradient=clip_gradient)

    def fit(self, X, labels, epochs, learning_rate=None, verbose=True, return_history=False):
        if learning_rate is not None:
            self.learning_rate = learning_rate

        history = []

        for epoch in range(epochs):
            predictions = self.forward(X)

            loss = self.calculate_loss(predictions, labels)
            history.append(loss['global'])

            self.backprop()

            self.optimize()

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f'Epoch {epoch:4d}/{epochs} - Loss: {loss["global"]:.6f}')

        if return_history:
            return history

    def predict(self, X):
        return self.forward(X)
