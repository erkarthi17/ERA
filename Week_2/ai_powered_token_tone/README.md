**🎫 AI Powered Ticket Tone**

This project is an AI-powered ticket classification system built with FastAPI and OpenAI.
It analyzes user-submitted problem statements, detects emotion and theme, assigns a priority level, and raises a ticket automatically.

**🚀 Features**

🔍 Tone Detection – Classifies user emotions (Critical, Urgent, High, Neutral, Low).

🏷 Theme Extraction – Identifies issue themes (Authentication Issue, System Crash, Connectivity Issue, UI/UX Problem, General Issue).

🎯 Priority Assignment – Maps tone → priority (P1 to P4).

🤖 OpenAI Integration – Uses GPT models for accurate analysis.

🔄 Fallback Logic – If OpenAI API fails uses internal rule-based analysis and responds back to User.

🖥 Frontend UI – Simple HTML/JS interface to submit problems and display ticket details.

**🛠 Tech Stack**

Backend: FastAPI, Uvicorn, Gunicorn

AI: OpenAI API (gpt-4 / gpt-3.5-turbo)

Frontend: HTML, CSS, JS

Other: Pydantic, dotenv

📦 **Installation in Local Machine**
1. Clone the repo
git clone [https://github.com/erkarthi17/ERA/Week_2/ai_powered_token_tone/](https://github.com/erkarthi17/ERA/tree/999d799eca63b688b1a3b5e29ac05530e53d4669/Week_2/ai_powered_token_tone)

2. Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows

4. Install dependencies
pip install -r requirements.txt

5. Add OpenAI API key

Create a .env file in the project root:

OPENAI_API_KEY=sk-your-api-key-here

**▶️ Running the App**
Start backend (FastAPI):
python uvicorn backend.app:app --reload
Start frontend (UI):
python -m http.server 8080

**Access frontend:**

Open index.html in your browser (frontend talks to FastAPI at http://127.0.0.1:8000).

**📊 API Endpoints**
POST /analyze

Analyzes a problem statement and returns a structured response.

Request:
{
  "problem": "All systems are down at the office, need help ASAP"
}

Response:
{
  "ticket_id": "TKT-8F92A1C3",
  "theme": "System Crash",
  "emotion": "Critical",
  "priority": "P1",
  "source": "OPENAI",
  "message": "We understand your problem. Considering the statement given, we have raised a P1 ticket.",
  "confidence": 0.95
}

**📌 Notes**

If OpenAI API fails, the system falls back to internal logic.

source field helps track whether analysis came from OPENAI or INTERNAL FALLBACK.

Tickets are assigned unique IDs with shortuuid.

Extend internal logic in model.py to fine-tune rule-based fallback.

**🎉Deployment Steps:**

1. Make sure that the Git Repo contains the latest code
2. Add a new '.env' file in the repo and update the OPENAI API Key
3. Create a new instance in the AWS portal (as per our Server requirement)
4. Make sure that the server instance is running fine without any issues
5. Upon validation, generate a key-pair (RSA) and use that key-pair value to connect to the public IP provided in the AWS Server instance
6. Use the command **ssh -i .\YOUROWNKEYPAIRFILE.pem ubuntu@YOURIPADDRESS**
7. This should initiate a successful communication with the server
8. Use git clone 'your repo' to clone the git hub repository in the AWS server instance
9. Followed by install necessary updates in the ubuntu os such as.,
     **sudo apt update
     sudo apt install python3 python3-pip**
10. Install necessary depedencies using below command.,
      pip3 install -r requirements.txt
11. start the backend using below.,
      **cd backend
      uvicorn app:app --host 0.0.0.0 --port 8000**
12. Launch a new terminal and start the frontend using below.,
      **cd frontend
      python3 -m http.server 8080**
13. Before launching the applications, make sure that Security group in the AWS has been amended in a way to allow the traffic to and from IP address
