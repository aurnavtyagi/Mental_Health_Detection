# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (if not already done)
1. Open `notebooks/mental_health_detector.ipynb` in Jupyter
2. Run all cells to train and save the model
3. Model files will be saved in `models/` directory

### Step 3: Run the Dashboard
```bash
# Windows
run_app.bat

# OR directly with Streamlit
streamlit run app.py

# Linux/Mac
./run_app.sh
# OR
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📱 Using the Dashboard

### Single Text Analysis
1. Click on "🔍 Single Text Analysis" tab
2. Enter or paste your text
3. Click "🔍 Analyze Text"
4. View results with probabilities and charts

### Batch Analysis
1. Click on "📁 Batch Upload" tab
2. Choose:
   - **CSV Upload**: Upload a CSV file with a 'text' column
   - **Multiple Texts**: Enter texts one per line
3. Click "🔍 Analyze All"
4. Download results as CSV

### Adjust Sensitivity
- Use the slider in the sidebar to adjust detection threshold
- Lower = more sensitive (catches more cases)
- Higher = more conservative (fewer false positives)

## 🎯 Example Usage

### Example 1: Single Tweet
```
Text: "I'm feeling really down today. Can't stop thinking about negative things."
Result: Depression detected (with probability percentage)
```

### Example 2: Batch CSV
Create a CSV file with a 'text' column:
```csv
text
"I'm feeling great today!"
"I can't find motivation to do anything"
"Had a wonderful day at the park"
```

Upload and analyze all at once!

## ⚠️ Troubleshooting

### Model Not Found Error
- Make sure you've trained the model first (Step 2)
- Check that `models/mental_health_model.pkl` exists
- Check that `models/tfidf_vectorizer.pkl` exists

### Port Already in Use
- Change the port: `streamlit run app.py --server.port 8502`
- Or stop the existing Streamlit process

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (requires 3.8+)

## 📊 Features

✅ Single text analysis with visualizations  
✅ Batch CSV upload and analysis  
✅ Multiple text input  
✅ Adjustable detection threshold  
✅ Probability charts and statistics  
✅ Download results as CSV  
✅ Analytics dashboard  

Enjoy using the Mental Health Detection Dashboard! 🧠

