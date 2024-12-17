# CMPE-257_SmartTrader_Team8

# SmartTrader: Machine Learning for Stock Price Prediction and Trading

## Team Members

**Member 1:**  
Name: Sai Pranavi Kurapati  
Email: saipranavi.kurapati@sjsu.edu  
Student ID: 017453904  

**Member 2:**  
Name: Prakhya Mylavaram  
Email: saiprakhya.mylavaram@sjsu.edu  
Student ID: 015999243  

**Member 3:**  
Name: Himaswetha Kurapati  
Email: himaswetha.kurapati@sjsu.edu  
Student ID: 017512534  

---

## Application URL
The application can be accessed at the following URL:

[**SmartTrader App**](https://smarttraderteam8final-0143352a003f.herokuapp.com/)

---

## Instructions to Access the Application

### 1. Accessing the App on Heroku
- Open the provided URL in a web browser.
- The app allows predictions for **NVIDIA Corporation (NVDA)** stock prices for the next **5 business days**.

### 2. Using the App
- Input a date in the **"Select Date"** field.
- Click on the **"Predict"** button to view the predicted high, low, and average prices, along with suggested trading strategies.

### 3. Expected Outputs
- A table displaying predictions for the next **5 business days**.
- A table showing trading strategies (**BULLISH, BEARISH, or IDLE**) based on the predictions.

### 4. Deployment Details
- Hosted on **Heroku** for real-time accessibility.
- Backend is built using **Flask** with machine learning models serialized and integrated for prediction.

---

## Build Instructions to Run Locally

### 1. Clone the Repository
```bash
   git clone https://github.com/SaiPranaviKurapati/CMPE-257_SmartTrader_Team8
   cd CMPE-257_SmartTrader_Team8
```

### 2. Set Up the Environment
Create a virtual environment:

**For Linux/Mac:**
```bash
   python3 -m venv venv
   source venv/bin/activate
```

**For Windows:**
```bash
   python -m venv venv
   venv\Scripts\activate
```

Install dependencies:
```bash
   pip install -r requirements.txt
```

### 3. Run the App Locally
Execute the following command:
```bash
   python app.py
```

Access the app locally at: [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Technologies Used
- **Backend:** Flask
- **Deployment:** Heroku
- **Machine Learning:** Integrated ML models for stock price prediction

---

## Contact
For questions or collaboration, please contact any of the team members via their provided emails.
