Note to self/
Overfitting: Try (Tried) dropout layers, L2 regularization, or early stopping.
Hyperparameter Tuning (Tried): Experiment with learning rate, LSTM units, and layers.
Data Scaling (tried): Ensure proper scaling, consider normalization.
Model Complexity: Adjust model complexity as needed.
Feature Engineering (not needed ig): Add more relevant data like technical indicators.

Performance Metrics
Loss Metrics
Training Loss: 41.3791
Validation Loss: 66.9168
Interpretation:
Ideally, these values should be close to each other. A higher validation loss compared to training loss might indicate some overfitting, meaning the model is performing well on training data but not generalizing well to unseen data (validation set).

Root Mean Squared Error (RMSE)
Test RMSE: 9.3605
Interpretation:
The lower the RMSE, the better the model performance. An RMSE of 9.3605 means that the model's predictions have an average deviation of about 9.36 units from the actual values.
Mean Absolute Error (MAE)
Training MAE: 5.6086
Validation MAE: 6.4538
Test MAE: 7.8533
Interpretation:
Lower MAE values indicate better model performance. Here, the model's average error on the training set is about 5.61 units, on the validation set it is about 6.45 units, and on the test set, it is about 7.85 units.
Mean Absolute Percentage Error (MAPE)
Training MAPE: 1.94%
Validation MAPE: 2.15%
Test MAPE: 2.73%
Interpretation:
Lower MAPE values indicate better model performance. A training MAPE of 1.94%, validation MAPE of 2.15%, and test MAPE of 2.73% means that, on average, the model's predictions are off by about 1.94%, 2.15%, and 2.73% respectively.

Training vs. Validation Performance:
The training loss is lower than the validation loss, which suggests some overfitting. However, the difference is not drastic, indicating that the model is somewhat generalizing but could be improved.
Test Performance:
The test RMSE of 9.3605 is relatively high, suggesting the model's predictions can deviate significantly from actual values.
The test MAE of 7.8533 and MAPE of 2.73% indicate that, on average, the model's predictions are off by around 7.85 units or 2.73%.
