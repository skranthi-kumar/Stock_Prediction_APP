# Stock Prediction App

## Overview
This project is a web-based stock prediction application that leverages machine learning models and sentiment analysis to predict stock prices based on historical data and market sentiment. The predictions are visualized using Power BI.

## Features
- Predicts future stock prices using LSTM models.
- Sentiment analysis on stock-related tweets.
- Visualization of stock data and predictions using interactive charts.

## Technologies
- **Backend**: Python, Flask, LSTM (Machine Learning)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Hadoop, Spark
- **Visualization**: Power BI

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/skranthi-kumar/Stock_Prediction_APP.git
```

### 2. Navigate to the project directory:
```bash
cd Stock_Prediction_APP
```

### 3. Set up a virtual environment:
It’s recommended to use a virtual environment to manage dependencies.

- Create a virtual environment:
  ```bash
  python -m venv venv
  ```

- Activate the virtual environment:

  - **Windows**:
    ```bash
    venv\Scripts\activate
    ```
  - **macOS/Linux**:
    ```bash
    source venv/bin/activate
    ```

### 4. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000` to access the application.

## File Structure
- `app.py`: Main Flask app script.
- `templates/`: HTML templates.
- `static/`: CSS and JS files for styling and interaction.
- `models/`: Machine learning models (LSTM).
- `data/`: Stock data used for predictions.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request.



This version includes the steps for setting up and activating a virtual environment. Let me know if anything else needs tweaking!
