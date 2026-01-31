# 💳 Credit Scoring Model

A complete machine learning application for credit risk assessment with a user-friendly web interface built with Streamlit.

## 🎯 Features

- **Instant Credit Score Predictions** - Assess creditworthiness in seconds
- **Interactive Web Interface** - User-friendly Streamlit dashboard
- **Model Training** - Train custom models with your own data
- **Performance Metrics** - Detailed model evaluation and analytics
- **Data Visualization** - Charts and graphs for better insights
- **Production-Ready** - Well-structured, modular codebase

## 📋 Project Structure

```
CodeAlpha_Credit-scoring-model/
├── app.py                  # Streamlit web application
├── model.py               # Original training script (reference)
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── src/
│   └── train_model.py    # Modular training module
├── models/               # Trained model storage
├── data/                 # Dataset storage
└── .gitignore           # Git configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Kaggle account (for dataset download)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adhars2006/CodeAlpha_Credit-scoring-model.git
   cd CodeAlpha_Credit-scoring-model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Kaggle API (for dataset download)**
   - Download your `kaggle.json` from [Kaggle Settings](https://www.kaggle.com/settings/account)
   - Place it in `~/.kaggle/`
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Usage

#### Option 1: Using the Web Interface (Recommended)

```bash
streamlit run app.py
```

This will open the web application in your browser with multiple pages:
- **Home** - Overview and current model performance
- **Predict Credit Score** - Make predictions on new customers
- **Train Model** - Train or retrain the model
- **About** - Project information

#### Option 2: Training from Command Line

```bash
python src/train_model.py
```

This will:
1. Download the dataset from Kaggle
2. Clean and preprocess the data
3. Train the Random Forest model
4. Display performance metrics
5. Save the model for predictions

## 📊 Model Details

### Algorithm
- **Type**: Random Forest Classifier
- **Trees**: 100
- **Train/Test Split**: 80/20
- **Cross-validation**: Random state = 42

### Features Used
The model uses 13+ features including:
- **Demographics**: Age, Occupation
- **Financial**: Annual Income, Outstanding Debt, Monthly Balance
- **Credit**: Credit History Age, Credit Limit, Credit Mix
- **Behavior**: Payment patterns, Delayed Payments, Bank Accounts
- **Accounts**: Number of loans, credit cards, bank accounts

### Performance Metrics
Latest model performance:
- **Precision**: 0.7936
- **Recall**: 0.7618
- **F1-Score**: 0.7774
- **ROC-AUC**: 0.9324

## 📈 Dataset

**Source**: [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

**License**: CC0-1.0 (Public Domain)

**Size**: ~100,000 records with 28 features

**Target Classes**:
- Good Credit Score
- Poor Credit Score

## 🛠️ Development

### Project Architecture

```python
CreditScoringModel
├── download_dataset()      # Fetch data from Kaggle
├── clean_data()           # Handle missing values, outliers
├── encode_data()          # Convert categorical features
├── train()                # Train Random Forest model
├── predict()              # Make predictions
├── save_model()           # Persist model
└── load_model()           # Load saved model
```

### Streamlit App Structure

```
Pages:
├── Home          - Dashboard with model overview
├── Predict       - Interactive prediction interface
├── Train         - Model training interface
└── About         - Project information
```

## 📝 Usage Examples

### Making a Prediction via Web UI

1. Go to "Predict Credit Score" page
2. Fill in customer information
3. Click "Predict Credit Score" button
4. View results with confidence scores

### Training a New Model

1. Go to "Train Model" page
2. Click "Start Training" button
3. Wait for the process to complete
4. Review performance metrics
5. Model automatically saved for predictions

### Programmatic Usage

```python
from src.train_model import CreditScoringModel

# Load trained model
model = CreditScoringModel()
model.load_model('./models/credit_model.pkl')

# Make a prediction
features = [35, 50000, 2, 0, 5000, 1000, 500, 10000, 60, 3, 2, 1, 1]
prediction, probability = model.predict(features)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probability}")
```

## 🔒 Security & Privacy

- All data is processed locally
- No data is sent to external servers (except Kaggle for dataset download)
- Model predictions are instant and private
- Kaggle API credentials required only for training

## 🐛 Troubleshooting

### Dataset Download Error
```
Error: "Model not found!"
Solution: Train the model first using the "Train Model" page
```

### Kaggle Authentication Error
```
Error: "403 - Forbidden"
Solution: Ensure kaggle.json is properly placed in ~/.kaggle/
```

### Out of Memory Error
```
Solution: Reduce dataset size or use a machine with more RAM
```

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **streamlit**: Web application framework
- **matplotlib**: Data visualization
- **kaggle**: Dataset download API

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy your repository

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**CodeAlpha Credit Scoring Model**

A production-ready machine learning project for credit risk assessment.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with web UI
- Random Forest model implementation
- Streamlit dashboard
- Model training capability
- Prediction interface

## 🎓 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Machine Learning Basics](https://developers.google.com/machine-learning/crash-course)
- [Credit Scoring Theory](https://en.wikipedia.org/wiki/Credit_score)

---

Made with ❤️ for credit risk assessment