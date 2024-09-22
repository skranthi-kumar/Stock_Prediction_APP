// Example JavaScript for Chart.js to render your stock data
const ctx = document.getElementById('stockChart').getContext('2d');
const stockChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['January', 'February', 'March', 'April', 'May'], // Replace with your data labels
        datasets: [{
            label: 'Stock Prices',
            data: [12, 19, 3, 5, 2], // Replace with your stock price data
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});
