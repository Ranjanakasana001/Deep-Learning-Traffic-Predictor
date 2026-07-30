import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, and MAPE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Avoid division by zero for MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return mae, rmse, mape

def evaluate_and_plot(model, test_loader, scaler, device='cpu', results_dir='../results/'):
    """
    Evaluates the model on the test set and plots predictions.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    # Concatenate all batches
    preds = np.concatenate(all_preds, axis=0) # Shape: [samples, pred_length, 1]
    targets = np.concatenate(all_targets, axis=0)
    
    # We will evaluate based on the first prediction step (e.g. t+1)
    # to keep visualization simple and interpretable
    preds_t1 = preds[:, 0, 0].reshape(-1, 1)
    targets_t1 = targets[:, 0, 0].reshape(-1, 1)
    
    # Inverse transform to get actual speed values
    preds_t1_inv = scaler.inverse_transform(preds_t1)
    targets_t1_inv = scaler.inverse_transform(targets_t1)
    
    # Calculate metrics
    mae, rmse, mape = calculate_metrics(targets_t1_inv, preds_t1_inv)
    print("\n--- Evaluation Metrics (t+1 prediction) ---")
    print(f"MAE:  {mae:.4f} km/h")
    print(f"RMSE: {rmse:.4f} km/h")
    print(f"MAPE: {mape:.4f} %")
    print("-------------------------------------------\n")
    
    # Plotting a subset of the test data (e.g., 500 time steps) to visualize clearly
    plot_len = min(500, len(targets_t1_inv))
    
    plt.figure(figsize=(15, 6))
    plt.plot(targets_t1_inv[:plot_len], label='Actual Traffic Speed', color='blue', alpha=0.7)
    plt.plot(preds_t1_inv[:plot_len], label='Predicted Speed (Transformer)', color='red', linestyle='--')
    
    plt.title("Traffic Congestion Forecasting: Actual vs Predicted Speed", fontsize=16)
    plt.xlabel("Time Steps (5-min intervals)", fontsize=12)
    plt.ylabel("Traffic Speed (km/h)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'prediction_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Evaluation plot saved to {plot_path}")
    
    # Write metrics to a file for the teacher
    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("--- Traffic Congestion Forecasting Results ---\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f} km/h\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f} km/h\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f} %\n")
        
    return mae, rmse, mape
