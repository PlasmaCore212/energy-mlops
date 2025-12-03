// Configuration
const API_URL = 'http://localhost:30080';  // Local Kubernetes NodePort

let predictionChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);
    loadStatistics();
});

// Handle prediction form submission
async function handlePrediction(e) {
    e.preventDefault();
    
    const hoursAhead = parseInt(document.getElementById('hoursAhead').value);
    const currentPower = document.getElementById('currentPower').value;
    const voltage = document.getElementById('voltage').value;
    const intensity = document.getElementById('intensity').value;
    
    const requestBody = {
        hours_ahead: hoursAhead
    };
    
    if (currentPower) requestBody.current_power = parseFloat(currentPower);
    if (voltage) requestBody.voltage = parseFloat(voltage);
    if (intensity) requestBody.intensity = parseFloat(intensity);
    
    showStatus('Making prediction...', 'loading');
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
        showStatus('Prediction successful!', 'success');
        
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

// Display prediction results
function displayResults(data) {
    const resultsInfo = document.getElementById('resultsInfo');
    resultsInfo.innerHTML = `
        <p><strong>Model Version:</strong> ${data.model_version}</p>
        <p><strong>Predictions:</strong> ${data.predictions.length} hours</p>
        <p><strong>Processing Time:</strong> ${data.prediction_time_ms.toFixed(2)}ms</p>
    `;
    
    // Prepare chart data
    const timestamps = data.predictions.map(p => {
        const date = new Date(p.timestamp);
        return date.toLocaleString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            hour: '2-digit' 
        });
    });
    
    const powers = data.predictions.map(p => p.predicted_power);
    
    // Create or update chart
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Predicted Power (kW)',
                data: powers,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Power Consumption (kW)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            }
        }
    });
}

// Load statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_URL}/predictions/stats`);
        
        if (!response.ok) {
            throw new Error('Could not load statistics');
        }
        
        const data = await response.json();
        displayStatistics(data);
        
    } catch (error) {
        document.getElementById('statsContainer').innerHTML = `
            <p style="color: #999;">Statistics unavailable (database not connected)</p>
        `;
    }
}

// Display statistics
function displayStatistics(data) {
    const container = document.getElementById('statsContainer');
    
    let html = `
        <div class="stat-card">
            <h3>Total Predictions</h3>
            <p>${data.total_predictions || 0}</p>
        </div>
    `;
    
    if (data.average_absolute_error) {
        html += `
            <div class="stat-card">
                <h3>Average Error</h3>
                <p>${data.average_absolute_error.toFixed(4)} kW</p>
            </div>
        `;
    }
    
    if (data.hourly_patterns && data.hourly_patterns.length > 0) {
        const avgPower = data.hourly_patterns.reduce((sum, h) => sum + parseFloat(h.avg_power), 0) / data.hourly_patterns.length;
        html += `
            <div class="stat-card">
                <h3>Average Power</h3>
                <p>${avgPower.toFixed(2)} kW</p>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// Show status message
function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            statusDiv.className = 'status';
            statusDiv.textContent = '';
        }, 5000);
    }
}