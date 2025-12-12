# Mental Health Detection System

A machine learning-based system for detecting mental health indicators (specifically depression) from text/tweets using Natural Language Processing and Logistic Regression.

## Features

- 🧠 **Machine Learning Model**: Trained Logistic Regression model with ~95.6% accuracy
- 📊 **Interactive Dashboard**: Streamlit-based web application for easy analysis
- 🔍 **Single Text Analysis**: Analyze individual tweets or texts
- 📁 **Batch Processing**: Upload CSV files or enter multiple texts for batch analysis
- 📈 **Analytics**: View trends and distributions of analysis results
- ⚙️ **Customizable Threshold**: Adjustable depression detection sensitivity


## Sample screenshots
<img width="1874" height="949" alt="Screenshot 2025-12-12 194019" src="https://github.com/user-attachments/assets/a68146ed-d9ee-4fd8-adc7-f23697e39181" />

## Project Structure

```
mental_health_detector/
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Processed datasets
├── models/                     # Saved model files
│   ├── mental_health_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   └── mental_health_detector.ipynb  # Model training notebook
├── app.py                     # Streamlit dashboard application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd mental_health_detector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already trained):
   - Open `notebooks/mental_health_detector.ipynb`
   - Run all cells to train and save the model
   - Model files will be saved in `models/` directory

## Usage

### Running the Streamlit Dashboard

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**:
   - The app will open in your default web browser
   - Default URL: `http://localhost:8501`

### Using the Dashboard

#### Single Text Analysis
1. Go to the "🔍 Single Text Analysis" tab
2. Enter or paste text in the text area
3. Click "🔍 Analyze Text"
4. View results with probabilities and visualizations

#### Batch Analysis
1. Go to the "📁 Batch Upload" tab
2. Choose between:
   - **CSV Upload**: Upload a CSV file with a 'text' or 'tweet' column
   - **Multiple Texts**: Enter multiple texts (one per line)
3. Click "🔍 Analyze All"
4. View summary statistics and download results

#### Analytics
- View analysis history
- Track trends over time
- Monitor probability distributions

### Adjusting Detection Sensitivity

In the sidebar, you can adjust the depression detection threshold:
- **Lower threshold (30-35%)**: More sensitive, catches more cases (may have false positives)
- **Default (40%)**: Balanced sensitivity
- **Higher threshold (45-50%)**: More conservative, fewer false positives (may miss some cases)

## Model Details

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: 5000 TF-IDF features with 1-2 gram ranges
- **Accuracy**: ~95.6% on test set
- **Training Data**: Reddit posts about depression
- **Preprocessing**: 
  - Lowercasing
  - URL removal
  - Punctuation removal
  - Stopword removal
  - Lemmatization

## Notebook Usage

The Jupyter notebook (`notebooks/mental_health_detector.ipynb`) provides:
- Complete data preprocessing pipeline
- Model training and evaluation
- Prediction functions for programmatic use
- Model saving and loading

### Using Prediction Functions in Python

```python
from notebooks.mental_health_detector import predict_mental_health, predict_batch

# Single prediction
result = predict_mental_health("I'm feeling really down today", depression_threshold=0.40)
print(result['label'])
print(result['depression_probability'])

# Batch prediction
texts = ["Text 1", "Text 2", "Text 3"]
results = predict_batch(texts, depression_threshold=0.40)
```

## Important Notes

⚠️ **Disclaimer**: This tool is for informational purposes only and should not replace professional mental health evaluation. If you're experiencing mental health issues, please consult with a healthcare professional.

- The model is trained on Reddit posts and may need fine-tuning for Twitter-specific language patterns
- Short texts may have lower accuracy due to fewer features
- Borderline cases (within 15% of 50/50 probability) are flagged for review

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

