# 🚀 Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Kaggle Credentials (Optional, for model training)
```bash
# Download kaggle.json from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Using the Application

### 🏠 Home Page
- View model overview
- See current model performance metrics
- Learn about the features

### 🔮 Predict Credit Score
- Enter customer information
- Get instant credit score prediction
- View confidence scores
- See financial profile analysis

### 🔧 Train Model
- Download dataset from Kaggle
- Train a new Random Forest model
- View training progress
- See model performance metrics

### ℹ️ About
- Project information
- Technology stack
- Dataset details

## Features

✨ **User-Friendly Interface** - Intuitive Streamlit dashboard
📊 **Real-time Predictions** - Instant credit score assessment
🎯 **Model Training** - Train custom models with your data
📈 **Performance Metrics** - Detailed evaluation and analytics
🔒 **Data Privacy** - All processing done locally

## Project Structure

```
├── app.py              # Main Streamlit application
├── src/
│   └── train_model.py  # ML model and training logic
├── models/             # Saved model storage
├── data/               # Dataset storage
└── requirements.txt    # Python dependencies
```

## Troubleshooting

**Problem**: "Model not found"
- **Solution**: Go to "Train Model" page and click "Start Training"

**Problem**: Kaggle download fails
- **Solution**: Check kaggle.json is properly placed in ~/.kaggle/

**Problem**: Port already in use
- **Solution**: Use `streamlit run app.py --server.port 8502`

## What's Next?

1. ✅ Explore the Home page
2. ✅ Train a model on the Train Model page
3. ✅ Make predictions on the Predict Credit Score page
4. ✅ Review performance metrics

Happy credit scoring! 🎉
