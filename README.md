# GIT Linux Final Project - Interactive Financial Dashboard of the Cac40

LINK : http://20.160.99.10:8501/home_page


This project is a **financial analytics and portfolio management web application** focused on the **CAC 40**, combining **market monitoring, quantitative analysis, and systematic portfolio construction**.  
It is designed to provide both **high-level market insights** and **advanced quantitative tools** through an interactive interface.

---

### 1) Interactive Dashboard

The interactive dashboard provides a **global overview of the CAC 40 market dynamics**.  
It includes key indicators such as returns and prices to have a quick look of the current market state.  
The dashboard is designed for **fast decision-making**, offering a clear and intuitive visualization of market conditions.

<img width="1793" height="870" alt="image" src="https://github.com/user-attachments/assets/f87797f9-30c9-49b5-b356-42822e9f64c9" />

---

### 2) Daily report directly on your inbox

Users can subscribe to an **automated daily report**, sent directly by email.  
<img width="492" height="311" alt="image" src="https://github.com/user-attachments/assets/ea9f7a14-ed26-4064-a487-30af640bf09c" />

This report summarizes the **main market movements** of the day, allowing users to stay informed without accessing the platform manually.

<img width="1501" height="848" alt="image" src="https://github.com/user-attachments/assets/f27ae331-40ff-4b28-a803-c00789eabf25" />

---

### 3) Single Asset Management

This section focuses on the **individual analysis of a single CAC 40 component**.  
It provides detailed historical price evolution, return profiles, volatility, key metrics and machine learning features to help the user to make a decision. It also feature a backtesting strategy to try well know allocation strats.

<img width="1805" height="889" alt="image" src="https://github.com/user-attachments/assets/8a5dd3a5-a8ec-4251-8e31-4d1c7b772eed" />

---

### 4) Backtesting of a strategy : Black & Litterman with momentum views

The platform includes a*portfolio backtesting module based on the **Black-Litterman framework enhanced with momentum views**. This strategy is a dense and complicated and was developped during Malo's Research project (you can check the github here : https://github.com/MaloBardin/Research-Project and the paper is in publication process).
This allows users to backtest the strategy, compare it to benchmark portfolios and other assets, and evaluate performance using risk-adjusted metrics such as drawdowns, volatility, and Sharpe ratios. It also feature a multi run backtesting to proove the robustness of the Black and Litterman allocation.
The associated python file (VVE.py) is very large and wasn't created for this project, just modified to work with EU data.
<img width="1778" height="815" alt="image" src="https://github.com/user-attachments/assets/5e4af14f-f03d-424a-8c57-7fed3c108472" />

---

### 5) Fully hosted on Microsoft Azure using Docker

The application is fully containerized and deployed using **Docker**, ensuring reproducibility, scalability, and ease of deployment.  
It is hosted on Microsoft Azure, providing a reliable cloud infrastructure suitable for production-level financial applications.

## Prerequisites

Before starting, ensure that **Docker** is installed and running on your machine.

- [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Installation

Clone the repository and navigate to the folder:

```bash
git clone [https://github.com/MaloBardin/GIT-Linux-Final-Project.git](https://github.com/MaloBardin/GIT-Linux-Final-Project.git)
cd GIT-Linux-Final-Project
```

Build the Docker image:

```bash
docker build -t streamlit-app .
```

---

## Usage

### Run the container

To run the application and see logs in the terminal:

```bash
docker run -p 8501:8501 streamlit-app
```

## Accessing the Application

Once the container is running, open your browser at the following address:

- **Local:** http://localhost:8501
- **Remote Server:** `http://<YOUR_SERVER_IP>:8501`
