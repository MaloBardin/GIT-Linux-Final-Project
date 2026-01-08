## Installation & Launch

### Prerequisites

- Docker installed (Linux / macOS / Windows)

---

### Installation v

# Clone the repository

```bash
git clone https://github.com/MaloBardin/GIT-Linux-Final-Project.git
cd GIT-Linux-Final-Project
```

# Build the Docker image

```bash
docker build --no-cache -t streamlit-app .
```

# Run the container

```bash
docker run -p 8501:8501 streamlit-app

```

# Once the container is running, you can access the application at:

http://<SERVER_IP>:8501
