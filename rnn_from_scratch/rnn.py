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
        self.version = 'v0.2-manytoone'

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

    def mse(self, predict, label):
        predict_flat = np.array(predict).flatten()
        label_flat = np.array(label).flatten()
        loss = np.mean((predict_flat - label_flat) ** 2)
        return loss

    def ce(self, predict, label):
        eps = 1e-12
        predict_flat = np.array(predict).flatten()
        label_flat = np.array(label).flatten()
        loss = -np.mean(label_flat * np.log(predict_flat + eps))
        return loss

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

    # Forward Pass (Many-to-One)
    def forward(self, X):
        X = np.array(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f'Input must be 2D array, got {X.ndim}D')

        self.input_data = X
        self.forward_result = {
            'pre_act': {'hidden_units': [], 'output': None},
            'post_act': {'hidden_units': [], 'output': None}
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

        # Calculate output ONLY from the last hidden state (Many-to-One)
        h_last = self.forward_result['post_act']['hidden_units'][-1]
        output = (self.weights['wy'] @ h_last) + self.bias['by']
        self.forward_result['pre_act']['output'] = output

        # Apply output activation
        y_activated = self.output_act[self.output_activation](output)
        self.forward_result['post_act']['output'] = y_activated

        return self.forward_result['post_act']['output']

    def calculate_loss(self, predict, label):
        # Calculate loss between prediction and label
        self.label = np.array(label, dtype=float)
        return self.cal_loss[self.loss](predict, self.label)

    # Backward Pass (Many-to-One)
    def backprop_find_gradient(self):
        n_timestep = len(self.forward_result['post_act']['hidden_units'])

        self.gradient['hidden_state'] = []

        # Calculate output gradient (only one output at the end)
        y_pred = self.forward_result['post_act']['output'].flatten()
        y_true = np.array(self.label).flatten()

        if self.loss == 'ce' and self.output_activation == 'softmax':
            grad_z = self.der_softmax_ce(y_pred, y_true)
        else:
            if callable(self.der_output_act[self.output_activation]):
                der_act = self.der_output_act[self.output_activation](y_pred)
            else:
                der_act = self.der_output_act[self.output_activation]

            der_loss = self.der_los[self.loss](y_pred, y_true)
            grad_z = der_act * der_loss

        self.gradient['z'] = grad_z

        # BPTT - Start from the last timestep
        dh_next = self.weights['wy'].T @ self.gradient['z']

        # Backward through time
        for t in range(n_timestep - 1, -1, -1):
            # Get hidden state at time t
            h_t = self.forward_result['post_act']['hidden_units'][t].flatten()

            # Apply activation derivative
            dh = dh_next * self.der_hidden_act[self.activation](h_t)

            # Store gradient
            self.gradient['hidden_state'].insert(0, dh)

            # Propagate gradient to previous timestep (if not first timestep)
            if t > 0:
                dh_next = self.weights['wh'].T @ dh
            else:
                dh_next = np.zeros(self.hidden_units)

    def backprop_find_gradient_weight(self):
        n_timestep = len(self.forward_result['post_act']['hidden_units'])

        # Initialize weight gradients
        self.gradient['wy'] = np.zeros_like(self.weights['wy'])
        self.gradient['wx'] = np.zeros_like(self.weights['wx'])
        self.gradient['wh'] = np.zeros_like(self.weights['wh'])

        # Gradient for Wy (output weights) - only from last hidden state
        h_last = self.forward_result['post_act']['hidden_units'][-1].flatten()
        grad_z = self.gradient['z'].reshape(-1, 1)
        self.gradient['wy'] = grad_z @ h_last.reshape(1, -1)

        # Gradient for Wx (input weights) - accumulated across all timesteps
        for t in range(n_timestep):
            x_t = self.input_data[t].reshape(-1, 1)
            grad_h_t = self.gradient['hidden_state'][t].reshape(-1, 1)
            self.gradient['wx'] += grad_h_t @ x_t.T

        # Gradient for Wh (recurrent weights) - accumulated across timesteps
        for t in range(1, n_timestep):
            h_prev = self.forward_result['post_act']['hidden_units'][t - 1].flatten()
            grad_h_t = self.gradient['hidden_state'][t].reshape(-1, 1)
            self.gradient['wh'] += grad_h_t @ h_prev.reshape(1, -1)

    def backprop_find_gradient_bias(self):
        n_timestep = len(self.forward_result['post_act']['hidden_units'])

        # Initialize bias gradients
        self.gradient['by'] = np.zeros_like(self.bias['by'])
        self.gradient['bh'] = np.zeros_like(self.bias['bh'])

        # Gradient for by (output bias) - only one gradient
        self.gradient['by'] = self.gradient['z'].reshape(-1, 1)

        # Gradient for bh (hidden bias) - accumulated across all timesteps
        for t in range(n_timestep):
            self.gradient['bh'] += self.gradient['hidden_state'][t].reshape(-1, 1)

    def backprop(self):
        self.gradient = {}

        # Calculate all gradients
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

    def train(self, X_train, y_train, epochs, learning_rate=None, verbose=True, return_history=False):
        """
        Train the RNN model
        
        Args:
            X_train: List of sequences, each sequence is shape (timesteps, features)
            y_train: List of labels, each label is shape (num_output,)
            epochs: Number of training epochs
            learning_rate: Learning rate (optional)
            verbose: Whether to print training progress
            return_history: Whether to return loss history
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate

        history = []

        for epoch in range(epochs):
            total_loss = 0
            
            # Train on each sequence
            for X, y in zip(X_train, y_train):
                # Forward pass
                prediction = self.forward(X)

                # Calculate loss
                loss = self.calculate_loss(prediction, y)
                total_loss += loss

                # Backward pass
                self.backprop()

                # Update weights
                self.optimize()

            # Average loss for this epoch
            avg_loss = total_loss / len(X_train)
            history.append(avg_loss)

            # Print progress
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f'Epoch {epoch:4d}/{epochs} - Loss: {avg_loss:.6f}')

        if return_history:
            return history

    def predict(self, X):
        return self.forward(X)

    def show_version(self):
      return self.version
