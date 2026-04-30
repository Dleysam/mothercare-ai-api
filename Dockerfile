# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /code

# 3. Install system dependencies (required for XGBoost and Pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy the rest of your application code and the .pkl model
# This includes app.py and your mothercare_model_v1.pkl
COPY . .

# 7. Expose the port Hugging Face uses (7860)
EXPOSE 7860

# 8. Command to run the FastAPI app using Uvicorn
# We use --host 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
