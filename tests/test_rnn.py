import numpy as np
import pytest
from rnn_from_scratch import RNNScratch


class TestRNNInitialization:
    def test_init_basic(self):
        rnn = RNNScratch(
            input_features=3,
            hidden_units=5,
            num_output=1,
            learning_rate=0.01
        )
        
        assert rnn.input_features == 3
        assert rnn.hidden_units == 5
        assert rnn.num_output == 1
        assert rnn.learning_rate == 0.01
    
    def test_weight_shapes(self):
        rnn = RNNScratch(
            input_features=3,
            hidden_units=5,
            num_output=2,
            learning_rate=0.01
        )
        
        assert rnn.weights['wx'].shape == (5, 3)  # (hidden, input)
        assert rnn.weights['wh'].shape == (5, 5)  # (hidden, hidden)
        assert rnn.weights['wy'].shape == (2, 5)  # (output, hidden)
        
    def test_bias_shapes(self):
        rnn = RNNScratch(
            input_features=3,
            hidden_units=5,
            num_output=2,
            learning_rate=0.01
        )
        
        assert rnn.bias['bh'].shape == (5, 1)
        assert rnn.bias['by'].shape == (2, 1)
    
    def test_invalid_activation(self):
        with pytest.raises(ValueError):
            RNNScratch(
                input_features=3,
                hidden_units=5,
                num_output=1,
                learning_rate=0.01,
                activation='invalid'
            )
    
    def test_invalid_loss(self):
        with pytest.raises(ValueError):
            RNNScratch(
                input_features=3,
                hidden_units=5,
                num_output=1,
                learning_rate=0.01,
                loss='invalid'
            )


class TestRNNForward:
    def test_forward_output_shape(self):
        rnn = RNNScratch(
            input_features=3,
            hidden_units=5,
            num_output=2,
            learning_rate=0.01
        )
        
        X = np.random.randn(10, 3)  # 10 timesteps, 3 features
        output = rnn.forward(X)
        
        assert len(output) == 10  # 10 timesteps
        assert output[0].shape == (2, 1)  # 2 outputs
    
    def test_forward_invalid_input(self):
        rnn = RNNScratch(
            input_features=3,
            hidden_units=5,
            num_output=1,
            learning_rate=0.01
        )
        
        X_invalid = np.random.randn(10)  # 1D array
        with pytest.raises(ValueError):
            rnn.forward(X_invalid)


class TestRNNTraining:
    def test_fit_basic(self):
        """Test basic training"""
        np.random.seed(42)
        
        rnn = RNNScratch(
            input_features=2,
            hidden_units=3,
            num_output=1,
            learning_rate=0.01,
            loss='mse',
            activation='tanh'
        )
        
        X = np.random.randn(5, 2)
        y = np.random.randn(5, 1)
        
        # Should not raise any errors
        rnn.fit(X, y, epochs=10, verbose=False)
    
    def test_fit_returns_history(self):
        """Test that return_history works"""
        np.random.seed(42)
        
        rnn = RNNScratch(
            input_features=2,
            hidden_units=3,
            num_output=1,
            learning_rate=0.01
        )
        
        X = np.random.randn(5, 2)
        y = np.random.randn(5, 1)
        
        history = rnn.fit(X, y, epochs=10, verbose=False, return_history=True)
        
        assert len(history) == 10
        assert all(isinstance(loss, (int, float)) for loss in history)
    
    def test_fit_loss_decreases(self):
        np.random.seed(42)
        
        # Simple problem: predict sum of inputs
        X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        y = X.sum(axis=1, keepdims=True)
        
        rnn = RNNScratch(
            input_features=2,
            hidden_units=5,
            num_output=1,
            learning_rate=0.01,
            loss='mse',
            activation='tanh',
            output_activation='linear'
        )
        
        history = rnn.fit(X, y, epochs=100, verbose=False, return_history=True)
        
        # Loss should generally decrease
        assert history[-1] < history[0]


class TestRNNPrediction:
    def test_predict_shape(self):
        rnn = RNNScratch(
            input_features=3,
            hidden_units=5,
            num_output=2,
            learning_rate=0.01
        )
        
        X = np.random.randn(7, 3)
        predictions = rnn.predict(X)
        
        assert len(predictions) == 7
        assert predictions[0].shape == (2, 1)


class TestActivationFunctions:
    def test_tanh(self):
        """Test tanh activation"""
        rnn = RNNScratch(
            input_features=2,
            hidden_units=3,
            num_output=1,
            learning_rate=0.01,
            activation='tanh'
        )
        
        x = np.array([0, 1, -1])
        result = rnn.tanh(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sigmoid(self):
        rnn = RNNScratch(
            input_features=2,
            hidden_units=3,
            num_output=1,
            learning_rate=0.01,
            activation='sigmoid'
        )
        
        x = np.array([0, 1, -1])
        result = rnn.sigmoid(x)
        assert np.all((result >= 0) & (result <= 1))
    
    def test_relu(self):
        rnn = RNNScratch(
            input_features=2,
            hidden_units=3,
            num_output=1,
            learning_rate=0.01,
            activation='relu'
        )
        
        x = np.array([-1, 0, 1, 2])
        result = rnn.relu(x)
        expected = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)


class TestLossFunctions:
    def test_mse_loss(self):
        """Test MSE loss calculation"""
        rnn = RNNScratch(
            input_features=2,
            hidden_units=3,
            num_output=1,
            learning_rate=0.01,
            loss='mse'
        )
        
        predictions = [np.array([[1.0]]), np.array([[2.0]])]
        labels = [np.array([[1.5]]), np.array([[2.5]])]
        
        loss_dict = rnn.mse(predictions, labels)
        
        assert 'local' in loss_dict
        assert 'global' in loss_dict
        assert len(loss_dict['local']) == 2
        assert loss_dict['global'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
