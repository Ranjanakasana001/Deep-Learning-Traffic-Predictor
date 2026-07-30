# Deep Learning-Based Urban Traffic Congestion Forecasting Using Transformer Models

## Overview
This repository contains the implementation for the final university project: **Deep Learning-Based Urban Traffic Congestion Forecasting Using Transformer Models**.

Accurate long-term traffic congestion prediction is a complex challenge due to temporal and spatial dependencies. This project investigates the application of advanced deep learning models, specifically Time-Series Transformers, to forecast urban traffic congestion. 

**This project includes a modern, interactive web dashboard, making it fully ready to be deployed on platforms like Render or Heroku.**

## Project Structure
- `data/`: Contains the traffic dataset (CSV).
- `src/`: 
  - `data_generator.py`: Generates realistic simulated traffic data.
  - `dataset.py`: PyTorch `Dataset` and `DataLoader`.
  - `model.py`: Core Time-Series Transformer architecture.
  - `train.py`: Training script.
  - `evaluate.py`: Evaluation metrics and plotting.
- `models/`: Saved PyTorch model weights (`.pth`).
- `templates/` & `static/`: HTML, CSS, and JS for the web dashboard.
- `app.py`: Flask web server exposing the prediction API and serving the UI.
- `main.py`: Script to train the model locally.
- `requirements.txt`: Python dependencies.
- `render.yaml` & `Procfile`: Configuration for easy deployment to Render/Heroku.

## Meaningful Outcome
This project delivers a **functional, end-to-end Machine Learning pipeline and Web Application**.
- **For the Teacher:** The codebase demonstrates advanced ML architecture (Transformers vs standard LSTMs), proper data normalization, rolling-window time-series generation, and clear evaluation metrics (MAE, RMSE, MAPE).
- **For the Presentation:** Instead of just showing terminal output, you can present a beautiful, interactive web dashboard that queries the trained PyTorch model in real-time.

## Setup Instructions (Local Development)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   First, run the training pipeline. This will generate the synthetic traffic data, train the Transformer, and save the weights.
   ```bash
   python main.py
   ```

3. **Start the Web Dashboard**
   Once trained, start the Flask server.
   ```bash
   python app.py
   ```
   Open your browser to `http://127.0.0.1:5000` to see the live dashboard!

## Web Deployment (Render)

This repository is pre-configured for deployment on [Render](https://render.com).
1. Push this code to a GitHub repository.
2. Go to Render dashboard and create a new **Web Service**.
3. Connect your GitHub repository.
4. Render will automatically detect the `render.yaml` and `Procfile`.
5. Deploy! The app will automatically build and start serving predictions.
