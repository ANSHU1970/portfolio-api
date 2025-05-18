# portfolio-api

🚀 How to Start the Portfolio API Server (Ubuntu)

1. Clone the Repository
   
git clone https://github.com/ANSHU1970/portfolio-api.git

cd portfolio-api


2. Create a Virtual Environment (Recommended)
   
python3 -m venv venv

source venv/bin/activate


3. Install Dependencies
   
pip install -r requirements.txt


4. Run the API Server
   
Assuming main.py contains the FastAPI app:

uvicorn main:app --host 0.0.0.0 --port 8000 --reload


The server will be available at: http://localhost:8000

API documentation: http://localhost:8000/docs
