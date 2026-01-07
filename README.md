TODO :

1. Linux deployment 
2. Adding ticker info page

3. Quant A : 
-Adding prediction (ARIMA and ML)
-Adding other strategies


4. QUANT B :
-Update the strategy
-Loading bar
-Display individual ticker's curves (checkbox to choose the tickers)

5. Bloomberg data retrieval : code to retrieve data on a CSV


Deployment : 

docker build --no-cache -t my-streamlit-app .
docker run -p 8501:8501 my-streamlit-app


