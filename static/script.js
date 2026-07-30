let trafficChart = null;

const initChart = () => {
    const ctx = document.getElementById('trafficChart').getContext('2d');
    
    Chart.defaults.color = '#4b5563';
    Chart.defaults.font.family = "'Helvetica Neue', Arial, sans-serif";
    
    trafficChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Historical Speed (km/h)',
                    data: [],
                    borderColor: '#059669', // Emerald
                    backgroundColor: 'rgba(5, 150, 105, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#059669',
                },
                {
                    label: 'Transformer Prediction',
                    data: [],
                    borderColor: '#d97706', // Amber
                    backgroundColor: 'rgba(217, 119, 6, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#d97706',
                    pointRadius: 5,
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#1f2937',
                    bodyColor: '#4b5563',
                    borderColor: '#e5e7eb',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    suggestedMin: 10,
                    suggestedMax: 80
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
};

const formatTime = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const updateDashboard = async () => {
    const btn = document.getElementById('refresh-btn');
    const loading = document.getElementById('loading');
    
    btn.disabled = true;
    btn.style.opacity = '0.5';
    loading.classList.remove('hidden');
    
    try {
        const response = await fetch('/api/predict');
        const data = await response.json();
        
        if (data.status === 'success') {
            const histTimes = data.history.timestamps.map(formatTime);
            const histValues = data.history.values;
            
            const predTimes = data.prediction.timestamps.map(formatTime);
            const predValues = data.prediction.values;
            
            // Update UI metrics
            document.getElementById('current-speed').innerText = `${histValues[histValues.length - 1].toFixed(1)} km/h`;
            document.getElementById('forecast-speed').innerText = `${predValues[0].toFixed(1)} km/h`;
            
            // Update Chart
            // To make the line continuous, we add the last historical point to the prediction array, 
            // but for simplicity in chart.js we pad with nulls
            const allLabels = [...histTimes, ...predTimes];
            
            const histData = [...histValues, ...Array(predValues.length).fill(null)];
            
            // Start prediction line exactly at the end of history
            const predData = [...Array(histValues.length - 1).fill(null), histValues[histValues.length - 1], ...predValues];
            
            trafficChart.data.labels = allLabels;
            trafficChart.data.datasets[0].data = histData;
            trafficChart.data.datasets[1].data = predData;
            
            trafficChart.update();
        } else {
            console.error('API Error:', data.message);
            alert('Failed to fetch predictions. Make sure model is trained.');
        }
    } catch (error) {
        console.error('Fetch error:', error);
    } finally {
        btn.disabled = false;
        btn.style.opacity = '1';
        loading.classList.add('hidden');
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initSimulatorChart();
    
    // Load initial data
    updateDashboard();
    
    document.getElementById('refresh-btn').addEventListener('click', updateDashboard);

    // Simulator slider
    const slider = document.getElementById('speed-slider');
    const display = document.getElementById('slider-value-display');
    let debounceTimer = null;

    slider.addEventListener('input', () => {
        const val = slider.value;
        display.innerText = `${val} km/h`;
        
        // Update the display color based on speed
        if (val >= 50) {
            display.style.color = '#059669';
        } else if (val >= 30) {
            display.style.color = '#d97706';
        } else {
            display.style.color = '#dc2626';
        }

        // Debounce: wait 300ms after user stops sliding before calling the API
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => runSimulation(val), 300);
    });

    // Run initial simulation at default value
    runSimulation(slider.value);
});

// ========= SIMULATOR =========
let simulatorChart = null;

const initSimulatorChart = () => {
    const ctx = document.getElementById('simulatorChart').getContext('2d');
    
    simulatorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Simulated History',
                    data: [],
                    borderColor: '#6b7280',
                    backgroundColor: 'rgba(107, 114, 128, 0.08)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 2,
                },
                {
                    label: 'AI Prediction (Next 15 min)',
                    data: [],
                    borderColor: '#dc2626',
                    backgroundColor: 'rgba(220, 38, 38, 0.08)',
                    borderWidth: 3,
                    borderDash: [6, 4],
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#dc2626',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#1f2937',
                    bodyColor: '#4b5563',
                    borderColor: '#e5e7eb',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                y: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    suggestedMin: 0,
                    suggestedMax: 80,
                    title: {
                        display: true,
                        text: 'Speed (km/h)',
                        color: '#6b7280'
                    }
                }
            },
            animation: {
                duration: 600,
                easing: 'easeOutCubic'
            }
        }
    });
};

const runSimulation = async (speed) => {
    try {
        const response = await fetch(`/api/simulate?speed=${speed}`);
        const data = await response.json();

        if (data.status === 'success') {
            const historyLen = data.history.length;
            const predLen = data.prediction.length;

            // Build time labels: -55min, -50min, ... , 0, +5min, +10min, +15min
            const labels = [];
            for (let i = 0; i < historyLen; i++) {
                labels.push(`-${(historyLen - 1 - i) * 5}m`);
            }
            for (let i = 0; i < predLen; i++) {
                labels.push(`+${(i + 1) * 5}m`);
            }

            // History dataset: values + nulls for prediction slots
            const histData = [...data.history.map(v => parseFloat(v.toFixed(1))), ...Array(predLen).fill(null)];

            // Prediction dataset: nulls + bridge from last history point + prediction values
            const predData = [...Array(historyLen - 1).fill(null), parseFloat(data.history[historyLen - 1].toFixed(1)), ...data.prediction.map(v => parseFloat(v.toFixed(1)))];

            simulatorChart.data.labels = labels;
            simulatorChart.data.datasets[0].data = histData;
            simulatorChart.data.datasets[1].data = predData;
            simulatorChart.update();

            // Update analysis panel
            const analysis = data.analysis;
            const congestionEl = document.getElementById('sim-congestion-level');
            congestionEl.innerText = analysis.congestion_level;
            congestionEl.style.color = analysis.congestion_color;

            document.getElementById('sim-avg-speed').innerText = `${analysis.avg_predicted_speed} km/h`;
            document.getElementById('sim-recovery').innerText = analysis.recovery_minutes === 0 ? 'None needed' : `~${analysis.recovery_minutes} min`;
            document.getElementById('sim-recommendation').innerText = analysis.recommendation;
        }
    } catch (error) {
        console.error('Simulation error:', error);
    }
};
