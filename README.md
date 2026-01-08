# GIT Linux Final Project - Streamlit App

This project contains a Dockerized Streamlit application.

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
