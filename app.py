from flask import Flask, render_template, jsonify
import torch
import numpy as np
import pandas as pd
import os
from src.model import TimeSeriesTransformer
from src.dataset import get_dataloaders
from src.data_generator import generate_synthetic_traffic_data

app = Flask(__name__)

# Global variables for model and scaler
MODEL = None
SCALER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_data():
    global MODEL, SCALER
    data_path = 'data/traffic_speed.csv'
    model_path = 'models/best_model.pth'
    
    # Ensure data exists (if deployed, we might need to generate it on first run or upload it)
    if not os.path.exists(data_path):
        generate_synthetic_traffic_data(data_path, num_days=60, interval_minutes=5)
        
    _, _, SCALER = get_dataloaders(csv_file=data_path, seq_length=12, pred_length=3, batch_size=32)
    
    MODEL = TimeSeriesTransformer(input_dim=1, d_model=32, n_heads=4, num_layers=2, dim_feedforward=128, pred_length=3)
    
    if os.path.exists(model_path):
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()
        print("Model loaded successfully.")
    else:
        print("WARNING: Model weights not found. Run main.py first to train the model. Using untrained weights for now.")

# Initialize when app starts
load_model_and_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict')
def predict():
    """
    Returns recent historical data and future predictions.
    """
    try:
        data_path = 'data/traffic_speed.csv'
        df = pd.read_csv(data_path)
        
        # Get the most recent 12 data points (1 hour of history)
        recent_data = df.tail(12)
        history_values = recent_data['speed'].values
        history_times = recent_data['timestamp'].tolist()
        
        # Scale and prepare tensor
        scaled_history = SCALER.transform(history_values.reshape(-1, 1))
        input_tensor = torch.tensor(scaled_history, dtype=torch.float32).unsqueeze(0).to(DEVICE) # Shape: [1, 12, 1]
        
        # Predict
        if MODEL:
            with torch.no_grad():
                pred_scaled = MODEL(input_tensor) # Shape: [1, 3, 1]
                
            pred_values = SCALER.inverse_transform(pred_scaled.squeeze().cpu().numpy().reshape(-1, 1)).flatten()
        else:
            # Fallback if no model
            pred_values = np.array([history_values[-1]] * 3)
            
        # Generate future timestamps
        last_time = pd.to_datetime(history_times[-1])
        future_times = [(last_time + pd.Timedelta(minutes=5 * (i + 1))).strftime("%Y-%m-%d %H:%M:%S") for i in range(3)]
        
        return jsonify({
            'status': 'success',
            'history': {
                'timestamps': history_times,
                'values': history_values.tolist()
            },
            'prediction': {
                'timestamps': future_times,
                'values': pred_values.tolist()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/simulate')
def simulate():
    """
    What-If Scenario Simulator.
    Takes a simulated current traffic speed from the slider and feeds a 
    synthetic history into the Transformer model to predict what happens next.
    """
    try:
        from flask import request
        simulated_speed = float(request.args.get('speed', 40))
        
        # Build a realistic 12-step history that gradually arrives at the simulated speed.
        # This gives the Transformer meaningful context instead of a flat line.
        base_speed = 60.0
        history = []
        for i in range(12):
            # Smooth transition from normal traffic to the simulated condition
            ratio = i / 11.0
            step_speed = base_speed + (simulated_speed - base_speed) * ratio
            # Add slight noise to look realistic
            step_speed += np.random.normal(0, 1.5)
            step_speed = max(5.0, min(75.0, step_speed))
            history.append(step_speed)
        
        history_arr = np.array(history).reshape(-1, 1)
        
        # Scale using the same scaler as training
        scaled_history = SCALER.transform(history_arr)
        input_tensor = torch.tensor(scaled_history, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Run Transformer inference
        if MODEL:
            with torch.no_grad():
                pred_scaled = MODEL(input_tensor)
            pred_values = SCALER.inverse_transform(pred_scaled.squeeze().cpu().numpy().reshape(-1, 1)).flatten().tolist()
        else:
            pred_values = [simulated_speed] * 3
        
        # Determine congestion level
        avg_predicted = np.mean(pred_values)
        if avg_predicted >= 50:
            congestion_level = "Low"
            congestion_color = "#059669"
            recommendation = "Traffic is flowing smoothly. No intervention needed."
        elif avg_predicted >= 35:
            congestion_level = "Moderate"
            congestion_color = "#d97706"
            recommendation = "Consider adjusting traffic signal timing to prevent buildup."
        elif avg_predicted >= 20:
            congestion_level = "High"
            congestion_color = "#dc2626"
            recommendation = "Activate alternate route suggestions. Extend green phases on main arterial."
        else:
            congestion_level = "Severe"
            congestion_color = "#7f1d1d"
            recommendation = "ALERT: Deploy traffic officers. Activate emergency rerouting protocols."
        
        # Estimate recovery time (simple heuristic based on how far from free-flow)
        free_flow = 65.0
        deficit = free_flow - avg_predicted
        recovery_minutes = max(0, int(deficit * 1.2))
        
        return jsonify({
            'status': 'success',
            'input_speed': simulated_speed,
            'history': history,
            'prediction': pred_values,
            'analysis': {
                'congestion_level': congestion_level,
                'congestion_color': congestion_color,
                'avg_predicted_speed': round(avg_predicted, 1),
                'recovery_minutes': recovery_minutes,
                'recommendation': recommendation
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
