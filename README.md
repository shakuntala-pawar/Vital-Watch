# Vital Watch – AI-Based Health Monitoring System

## Overview

**Vital Watch** is an intelligent health monitoring system designed to track patient vital signs and provide early risk predictions using machine learning. The system enables real-time monitoring, data analysis, and alert generation to support proactive healthcare decisions.

This project demonstrates the integration of **Artificial Intelligence, backend services, and a user-friendly web interface** to simulate a modern digital health monitoring solution suitable for hospitals, clinics, and remote patient monitoring scenarios.

---

## Key Features

* Real-time patient vital monitoring
* Machine learning–based risk prediction
* Patient data management using CSV datasets
* REST API for backend communication
* Interactive web-based frontend interface
* Scalable backend architecture
* Health risk classification and alert logic
* Modular design for future expansion

---

## System Architecture

The system follows a simple full-stack architecture:

Frontend (HTML Interface)
⬇
Backend Server (FastAPI)
⬇
Machine Learning Engine (Prediction Model)
⬇
Patient Dataset (CSV)

---

## Tech Stack

### Programming Languages

* Python
* HTML
* JavaScript (basic frontend logic)

### Frameworks and Libraries

* FastAPI – Backend API framework
* Uvicorn – ASGI server
* Pandas – Data processing
* NumPy – Numerical computations
* Scikit-learn – Machine learning model

### Tools and Platforms

* Git and GitHub – Version control
* VS Code – Development environment
* Windows PowerShell – Command-line interface

---

## Project Structure

```
Vital-Watch/
│
├── backend/
│   ├── server.py          # FastAPI backend server
│   ├── ml_engine.py       # Machine learning prediction logic
│
├── frontend/
│   ├── index.html         # User interface
│
├── sample_patients.csv     # Sample patient dataset
├── run.py                  # Application entry point
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Installation and Setup

Follow these steps to run the project locally.

### Step 1 – Clone the Repository

```
git clone https://github.com/shakuntala-pawar/Vital-Watch.git
cd Vital-Watch
```

### Step 2 – Create Virtual Environment

```
python -m venv .venv
```

### Step 3 – Activate Virtual Environment

**Windows:**

```
.venv\Scripts\activate
```

### Step 4 – Install Dependencies

```
pip install -r requirements.txt
```

### Step 5 – Run the Application

```
python run.py
```

The server will start locally and the application can be accessed through the browser.

---

## How It Works

1. Patient data is entered through the frontend interface.
2. The backend API receives the data using FastAPI.
3. The machine learning model processes the patient vitals.
4. The system predicts the health risk level.
5. The result is displayed to the user.

---

## Sample Use Cases

* Remote patient health monitoring
* Early detection of health risks
* Hospital patient tracking systems
* AI-assisted healthcare analytics
* Academic and research demonstrations

---

## Future Enhancements

* Real-time sensor integration (IoT devices)
* Database integration (MySQL / PostgreSQL)
* User authentication and role management
* Deployment to cloud platforms
* Data visualization dashboards
* Mobile application support
* Advanced deep learning prediction models

---

## Skills Demonstrated

* Full-stack application development
* Machine learning model integration
* API development using FastAPI

