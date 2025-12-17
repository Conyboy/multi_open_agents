
FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js for Studio
RUN npm install -g openagents-studio --prefix /usr/local

# Copy all code
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose ports: network (8700), studio (8050), HTTP (8600)
EXPOSE 8700 8600 8050

ENTRYPOINT ["/app/entrypoint.sh"]
