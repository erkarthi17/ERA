#!/bin/bash

# Set IP address (can be EC2 private IP or 0.0.0.0 for all interfaces)
HOST_IP="0.0.0.0"

# Start Backend (FastAPI via Uvicorn)
echo "🚀 Starting FastAPI backend on $HOST_IP:5000..."
cd backend
nohup python3 -m uvicorn app:app --host $HOST_IP --port 5000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Start Frontend (Static HTML/CSS/JS via Python HTTP server)
echo "🎨 Starting frontend on $HOST_IP:8080..."
cd frontend
nohup python3 -m http.server 8080 --bind $HOST_IP > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "✅ Both services started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Access backend at: http://<your-ec2-public-ip>:5000"
echo "Access frontend at: http://<your-ec2-public-ip>:8080"
