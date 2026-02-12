document.getElementById('sensorForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // 1. Collect form data
    const formData = new FormData(e.target);
    const sensorData = Object.fromEntries(formData.entries());

    // 2. UI Loading State
    const gauge = document.getElementById('gauge');
    const statusLabel = document.getElementById('statusLabel');
    gauge.innerText = "loading...";

    try {
        // 3. Send to Flask API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sensorData)
        });

        const result = await response.json();

        // 4. Update UI with result
        gauge.innerText = result.probability + "%";
        statusLabel.innerText = "Status: " + result.status;

        // Dynamic Color Logic
        if (result.probability > 70) {
            gauge.style.color = "#ef4444"; // Red
            statusLabel.style.color = "#ef4444";
        } else {
            gauge.style.color = "#22c55e"; // Green
            statusLabel.style.color = "#22c55e";
        }

    } catch (error) {
        console.error("Prediction failed:", error);
        gauge.innerText = "ERR";
    }
});