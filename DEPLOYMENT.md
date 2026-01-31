# 📦 Deployment Guide

This guide covers how to deploy the Credit Scoring Model application in different environments.

## Local Development

### Prerequisites
- Python 3.8+
- pip or conda
- Kaggle account (optional, for training)

### Steps

1. **Clone and setup**
   ```bash
   git clone https://github.com/Adhars2006/CodeAlpha_Credit-scoring-model.git
   cd CodeAlpha_Credit-scoring-model
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   streamlit run app.py
   ```

3. **Access the app**
   - Open browser: `http://localhost:8501`

## Docker Deployment

### Build Docker Image

```dockerfile
# Save as Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the app
CMD ["streamlit", "run", "app.py"]
```

### Build and Run

```bash
# Build image
docker build -t credit-scoring:latest .

# Run container
docker run -p 8501:8501 \
  -v ~/.kaggle:/root/.kaggle \
  credit-scoring:latest
```

## Cloud Deployment (Streamlit Cloud)

### Steps

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Create Streamlit Cloud account**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy application**
   - Click "New app"
   - Select repository: `CodeAlpha_Credit-scoring-model`
   - Select branch: `main`
   - Set main file: `app.py`
   - Click "Deploy"

4. **Configure secrets** (if needed)
   - Go to app settings
   - Add environment variables or Kaggle credentials

## Heroku Deployment

### Prerequisites
- Heroku CLI installed
- Heroku account

### Steps

1. **Create Procfile**
   ```
   web: sh setup.sh && streamlit run app.py
   ```

2. **Create setup.sh**
   ```bash
   #!/bin/bash
   mkdir -p ~/.streamlit/
   echo "[server]
   port = $PORT
   enableXsrfProtection = false
   headless = true
   " > ~/.streamlit/config.toml
   ```

3. **Deploy**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

## AWS EC2 Deployment

### Steps

1. **Launch EC2 instance**
   - Ubuntu 20.04 LTS
   - t2.micro (free tier)
   - Security group: Allow HTTP (80), HTTPS (443), SSH (22)

2. **SSH into instance**
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-ip
   ```

3. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   ```

4. **Clone and setup**
   ```bash
   git clone https://github.com/Adhars2006/CodeAlpha_Credit-scoring-model.git
   cd CodeAlpha_Credit-scoring-model
   pip install -r requirements.txt
   ```

5. **Run with systemd**
   - Create service file: `/etc/systemd/system/credit-scoring.service`
   ```ini
   [Unit]
   Description=Credit Scoring App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/CodeAlpha_Credit-scoring-model
   ExecStart=/usr/bin/streamlit run app.py --server.port=80
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
   
   - Enable and start:
   ```bash
   sudo systemctl enable credit-scoring
   sudo systemctl start credit-scoring
   ```

## Google Cloud Run Deployment

### Prerequisites
- Google Cloud account
- Google Cloud SDK

### Steps

1. **Create app.yaml**
   ```yaml
   runtime: python39
   entrypoint: streamlit run app.py --logger.level=error --server.address=0.0.0.0
   
   env: standard
   threadsafe: true
   
   handlers:
   - url: /.*
     script: auto
   ```

2. **Deploy**
   ```bash
   gcloud app deploy
   ```

## Azure App Service Deployment

### Steps

1. **Create web app**
   ```bash
   az webapp create \
     --resource-group myResourceGroup \
     --plan myPlan \
     --name credit-scoring-app \
     --runtime "PYTHON|3.9"
   ```

2. **Deploy code**
   ```bash
   az webapp up \
     --resource-group myResourceGroup \
     --name credit-scoring-app \
     --runtime "PYTHON|3.9"
   ```

## Environment Variables

Create a `.env` file for configuration:

```env
# Kaggle credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Streamlit settings
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501

# Application settings
APP_DEBUG=false
```

## Performance Optimization

### For Production

1. **Cache models**
   ```python
   @st.cache_resource
   def load_model():
       # Model loading logic
   ```

2. **Use CDN for assets**
   - Store static files in S3 or CloudFront

3. **Add load balancer**
   - Use Nginx or AWS ELB
   - Distribute traffic across instances

4. **Database caching**
   - Cache predictions in database
   - Improve response times

## Monitoring & Logging

### Streamlit Cloud
- Built-in logs available in app settings
- Monitor app performance

### Docker/Self-hosted
- Use ELK Stack (Elasticsearch, Logstash, Kibana)
- Or CloudWatch, DataDog, New Relic

### Log Commands
```bash
# View container logs
docker logs -f credit-scoring

# System logs (if running as service)
sudo journalctl -u credit-scoring -f
```

## Security Best Practices

1. **Use HTTPS**
   - Install SSL certificate
   - Redirect HTTP to HTTPS

2. **Secure Kaggle credentials**
   - Use environment variables
   - Never commit secrets to Git

3. **Rate limiting**
   - Implement API rate limiting
   - Use web server WAF

4. **Input validation**
   - Validate all user inputs
   - Prevent injection attacks

5. **Keep dependencies updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## Backup & Restore

### Backup trained models
```bash
# Copy models directory
cp -r models/ backup/models/

# Create archive
tar -czf credit_models_backup.tar.gz models/
```

### Database backup (if using)
```bash
# PostgreSQL example
pg_dump database_name > backup.sql

# Restore
psql database_name < backup.sql
```

## Troubleshooting

### Port already in use
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run app.py --server.port 8502
```

### Memory issues
```bash
# Limit memory usage
docker run -m 512m credit-scoring:latest
```

### Kaggle API errors
- Verify kaggle.json is in ~/.kaggle/
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Regenerate API key if necessary

## Support

For deployment issues, please open an issue on GitHub with:
- Deployment platform used
- Error messages
- Steps to reproduce
- System information
