
# 1. Base Image
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy the requirements first (for efficient caching)
COPY requirements.txt .

# 4. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy script code into image
COPY . .

# 5. Specify the Command to Run the Application
CMD ["python", "simple_ml_script.py"]


