import os
import torch
from src.data_generator import generate_synthetic_traffic_data
from src.dataset import get_dataloaders
from src.model import TimeSeriesTransformer
from src.train import train_model
from src.evaluate import evaluate_and_plot

def main():
    print("=== Deep Learning-Based Urban Traffic Congestion Forecasting ===")
    print("Project Initialization...\n")
    
    # 1. Configuration
    data_path = 'data/traffic_speed.csv'
    model_path = 'models/best_model.pth'
    seq_length = 12   # Past 1 hour (12 * 5 mins)
    pred_length = 3   # Predict next 15 mins (3 * 5 mins)
    epochs = 15       # Keeping it relatively short for demonstration
    batch_size = 64
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Data Generation (if not exists)
    if not os.path.exists(data_path):
        print("Data not found. Generating realistic synthetic traffic data...")
        # 60 days to give the transformer enough data to learn the pattern
        generate_synthetic_traffic_data(data_path, num_days=60, interval_minutes=5)
    else:
        print(f"Dataset found at {data_path}.")
        
    # 3. Data Loading Pipeline
    print("\nInitializing Data Pipeline...")
    train_loader, test_loader, scaler = get_dataloaders(
        csv_file=data_path,
        seq_length=seq_length,
        pred_length=pred_length,
        batch_size=batch_size
    )
    print(f"Training batches: {len(train_loader)} | Testing batches: {len(test_loader)}")
    
    # 4. Model Initialization
    print("\nInitializing Transformer Model...")
    model = TimeSeriesTransformer(
        input_dim=1,          # Only speed feature
        d_model=32,           # Reduced for faster CPU training in demo
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        pred_length=pred_length
    )
    
    # 5. Training
    print("\nStarting Training Phase...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        model_path=model_path
    )
    
    # 6. Evaluation and Visualization
    print("\nStarting Evaluation Phase...")
    # Load best weights
    trained_model.load_state_dict(torch.load(model_path, map_location=device))
    
    evaluate_and_plot(
        model=trained_model,
        test_loader=test_loader,
        scaler=scaler,
        device=device,
        results_dir='results/'
    )
    
    print("\n=== Project Execution Completed Successfully ===")
    print("Check the 'results/' folder for evaluation metrics and plots.")

if __name__ == '__main__':
    main()
