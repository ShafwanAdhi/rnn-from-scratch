import numpy as np
from rnn_from_scratch import RNNScratch

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic sequential data
# Input: 10 timesteps, 3 features
# Output: 10 timesteps, 1 value
X = np.array([
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6],
    [0.5, 0.6, 0.7],
    [0.6, 0.7, 0.8],
    [0.7, 0.8, 0.9],
    [0.8, 0.9, 1.0],
    [0.9, 1.0, 1.1],
    [1.0, 1.1, 1.2]
])

# Target: sum of features
y = X.sum(axis=1, keepdims=True)

print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("First 3 samples:")
print("X:", X[:3])
print("y:", y[:3])

# Initialize RNN
print("Training RNN for Regression")

rnn = RNNScratch(
    input_features=3,
    hidden_units=5,
    num_output=1,
    learning_rate=0.001,
    loss='mse',
    activation='linear',
    output_activation='linear'
)

# Train the model
history = rnn.fit(X, y, epochs=50000, learning_rate=0.00001, verbose=True, return_history=True)

# Make predictions
print("Predictions vs Actual")

predictions = rnn.predict(X)

for i, (pred, actual) in enumerate(zip(predictions, y)):
    pred_val = pred.flatten()[0]
    actual_val = actual.flatten()[0]
    error = abs(pred_val - actual_val)
    print(f"Timestep {i}: Predicted={pred_val:.4f}, Actual={actual_val:.4f}, Error={error:.4f}")

# Calculate final metrics
final_loss = history[-1]
print(f"Final MSE Loss: {final_loss:.6f}")

# Test on new data
print("Testing on New Data")

X_test = np.array([
    [1.1, 1.2, 1.3],
    [1.2, 1.3, 1.4],
    [1.3, 1.4, 1.5]
])

y_test = X_test.sum(axis=1, keepdims=True)
predictions_test = rnn.predict(X_test)

print("\nTest Results:")
for i, (pred, actual) in enumerate(zip(predictions_test, y_test)):
    pred_val = pred.flatten()[0]
    actual_val = actual.flatten()[0]
    error = abs(pred_val - actual_val)
    print(f"Sample {i}: Predicted={pred_val:.4f}, Actual={actual_val:.4f}, Error={error:.4f}")
