import numpy as np
import pandas as pd
import os

def generate_synthetic_traffic_data(file_path: str, num_days: int = 30, interval_minutes: int = 5):
    """
    Generates synthetic traffic speed data mimicking urban congestion patterns.
    
    The data simulates a typical daily commute pattern with morning and evening rush hours,
    lower speeds during rush hour (congestion), and higher speeds otherwise.
    
    Args:
        file_path (str): The path to save the generated CSV file.
        num_days (int): Number of days of data to generate.
        interval_minutes (int): Frequency of data collection (e.g., every 5 mins).
    """
    print(f"Generating synthetic traffic data for {num_days} days...")
    
    # Calculate total time steps
    steps_per_day = (24 * 60) // interval_minutes
    total_steps = num_days * steps_per_day
    
    # Create time index
    time_index = pd.date_range(start="2026-01-01", periods=total_steps, freq=f"{interval_minutes}min")
    
    # Base speed (free flow)
    base_speed = 65.0 # km/h
    
    # Generate patterns
    speeds = []
    for t in time_index:
        hour = t.hour
        day_of_week = t.dayofweek
        
        speed = base_speed
        
        # Weekday patterns
        if day_of_week < 5: 
            # Morning rush (7 AM - 9 AM)
            if 7 <= hour <= 9:
                speed -= np.random.uniform(20, 40)
            # Evening rush (4 PM - 7 PM)
            elif 16 <= hour <= 19:
                speed -= np.random.uniform(25, 45)
            # Daytime (9 AM - 4 PM)
            elif 9 < hour < 16:
                speed -= np.random.uniform(5, 15)
        # Weekend patterns
        else:
            # Mild midday congestion
            if 11 <= hour <= 15:
                speed -= np.random.uniform(10, 20)
                
        # Add random noise
        noise = np.random.normal(0, 3)
        speed = max(5.0, min(base_speed + noise, speed + noise)) # Keep speed within realistic bounds [5, base_speed+noise]
        
        speeds.append(speed)
        
    df = pd.DataFrame({
        'timestamp': time_index,
        'speed': speeds
    })
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Data successfully generated and saved to {file_path}")
    print(f"Total records: {len(df)}")
    
if __name__ == '__main__':
    generate_synthetic_traffic_data('../data/traffic_speed.csv', num_days=30)
