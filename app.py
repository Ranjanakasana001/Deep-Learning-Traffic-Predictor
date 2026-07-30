from flask import Flask, render_template, jsonify, request
import numpy as np
import os
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict')
def predict():
    """
    Returns realistic synthetic traffic data.
    This guarantees the graph always populates and never fails on cloud servers.
    """
    try:
        now = datetime.now()
        
        # Generate 12 realistic historical data points ending right now
        base_speed = 45.0
        history_values = []
        history_times = []
        for i in range(12, 0, -1):
            t = now - timedelta(minutes=5 * i)
            history_times.append(t.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Add some realistic noise
            speed = base_speed + np.random.normal(0, 4)
            history_values.append(max(10.0, min(80.0, speed)))
            
        # Generate 3 prediction points
        pred_values = []
        pred_times = []
        last_speed = history_values[-1]
        
        for i in range(1, 4):
            t = now + timedelta(minutes=5 * i)
            pred_times.append(t.strftime("%Y-%m-%d %H:%M:%S"))
            
            # The AI prediction mock: slightly converging back to base speed
            trend = (base_speed - last_speed) * 0.2
            next_speed = last_speed + trend + np.random.normal(0, 2)
            pred_values.append(max(10.0, min(80.0, next_speed)))
            last_speed = next_speed

        return jsonify({
            'status': 'success',
            'history': {
                'timestamps': history_times,
                'values': history_values
            },
            'prediction': {
                'timestamps': pred_times,
                'values': pred_values
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/simulate')
def simulate():
    """
    What-If Scenario Simulator.
    Takes a simulated current traffic speed from the slider and generates
    a synthetic history and realistic future prediction.
    """
    try:
        simulated_speed = float(request.args.get('speed', 40))
        
        # Build a realistic 12-step history that gradually arrives at the simulated speed.
        base_speed = 60.0
        history = []
        for i in range(12):
            ratio = i / 11.0
            step_speed = base_speed + (simulated_speed - base_speed) * ratio
            step_speed += np.random.normal(0, 1.5)
            history.append(max(5.0, min(75.0, step_speed)))
            
        # Prediction
        pred_values = []
        last_speed = simulated_speed
        for i in range(3):
            # Gradually recover towards base speed if congested, or stay free flow
            recovery_trend = (base_speed - last_speed) * 0.3
            next_speed = last_speed + recovery_trend + np.random.normal(0, 1.5)
            pred_values.append(max(5.0, min(80.0, next_speed)))
            last_speed = next_speed
        
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
            recommendation = "Activate alternate route suggestions. Extend green phases on MG Road."
        else:
            congestion_level = "Severe"
            congestion_color = "#7f1d1d"
            recommendation = "ALERT: Deploy traffic officers. Activate emergency rerouting protocols."
        
        # Estimate recovery time
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
