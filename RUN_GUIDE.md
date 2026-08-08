# 🚀 SignBridge Running Guide

This guide describes how to activate your environment and execute the **SignBridge (Indian Sign Language to Speech)** project on Windows.

The project offers two user interface dashboards:
1. **Option A (Recommended):** The premium, glassmorphic React frontend connected to a FastAPI WebSocket server (supporting low-latency multi-hand tracking, flow control, and dynamic theme settings).
2. **Option B (Legacy):** The simple Streamlit web application.

---

## 📋 Quick Start Steps

### Step 1: Open Terminal
Open your terminal and navigate to the project directory:
```powershell
cd "c:\Project shri 2026\Capstone-NVIDIA"
```

### Step 2: Activate the Virtual Environment
Activate the Python virtual environment:
* **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **In Command Prompt (cmd):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

> [!TIP]
> **PowerShell Execution Policy Error?**  
> If Windows blocks script execution with a red error message:
> 1. Run this command to temporarily bypass execution policies for this session:
>    ```powershell
>    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
>    ```
> 2. Then try running `.\venv\Scripts\Activate.ps1` again.

---

## 🏃 Running Option A: React + FastAPI WebSocket (Premium App)

To run the full real-time neural network tracking, you will launch both the python server and the React client:

### 1. Launch the FastAPI WebSocket Server
In your first activated terminal tab, start the backend server:
```powershell
python backend_server.py
```
*Wait for the console to output:* `INFO: Uvicorn running on http://0.0.0.0:8000`

### 2. Launch the React Client
Open a second terminal, navigate to the `frontend/` directory, and start the development server:
```powershell
cd frontend
npm run dev
```
*Open:* **[http://localhost:5173/](http://localhost:5173/)** in your browser.

### 3. Usage
- Go to the **Live Translation** page.
- Click **Start Detection** to open your webcam stream and connect to the WebSocket backend.
- Adjust parameters (e.g. Confidence Threshold, Stability Frames, accent voice, rate, and theme styles) on the **Settings** page.

---

## 🏃 Running Option B: Streamlit Web App (Legacy App)

If you wish to run the legacy Streamlit interface:
```powershell
streamlit run app.py
```
*(Alternatively, run it directly through the virtual environment's python executor)*
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
Open the page at:  
👉 **`http://localhost:8501`**

---

## 🧪 Testing Single Images (CLI)
You can run classification on any static test image in the dataset to verify that the neural network model is working:
```powershell
python predict.py dataset/a/0.jpg
```
The console will display the predicted alphabet class and the model's confidence rating.
