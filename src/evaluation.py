import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    train_mape = mean_absolute_percentage_error(y_train, train_predictions)
    val_mape = mean_absolute_percentage_error(y_val, val_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions)
    
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"Training MAPE: {train_mape:.2%}, Validation MAPE: {val_mape:.2%}, Test MAPE: {test_mape:.2%}")
    
    return train_predictions, val_predictions, test_predictions

def plot_predictions(dates, predictions, actuals, title):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, predictions, label='Predictions')
    plt.plot(dates, actuals, label='Actual Values')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
