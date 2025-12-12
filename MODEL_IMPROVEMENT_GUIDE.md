# Model Improvement Guide

## Overview

This guide explains the techniques used to improve model accuracy from ~95.6% to potentially **96-98%+** accuracy.

## Techniques Implemented

### 1. **Enhanced Feature Engineering**
- **Increased features**: 5000 → 8000 TF-IDF features
- **Better filtering**: 
  - `min_df=2`: Removes rare terms (appear in <2 documents)
  - `max_df=0.95`: Removes common terms (appear in >95% documents)
- **Sublinear TF scaling**: Uses `1 + log(tf)` instead of raw term frequency
- **Result**: Better feature quality, less noise

### 2. **Feature Selection**
- Uses Chi-square test to select top 6000 most informative features
- Removes noise and irrelevant features
- **Expected improvement**: +0.5-1% accuracy

### 3. **Multiple Model Comparison**
Tests 6 different algorithms:
- Logistic Regression (baseline)
- Random Forest
- **XGBoost** (usually best for text classification)
- SVM
- Gradient Boosting
- Naive Bayes

### 4. **Hyperparameter Tuning**
- Uses GridSearchCV to find optimal parameters
- Tunes: n_estimators, max_depth, learning_rate, subsample
- **Expected improvement**: +1-2% accuracy

### 5. **Ensemble Methods**
- Voting Classifier combines best models
- Uses soft voting (probability-based)
- Combines strengths of multiple algorithms
- **Expected improvement**: +0.5-1.5% accuracy

### 6. **Cross-Validation**
- 5-fold stratified cross-validation
- More robust evaluation
- Prevents overfitting

## Expected Results

| Technique | Expected Accuracy Gain |
|-----------|----------------------|
| Enhanced Features | +0.5-1% |
| Feature Selection | +0.5-1% |
| XGBoost vs LR | +1-2% |
| Hyperparameter Tuning | +1-2% |
| Ensemble | +0.5-1.5% |
| **Total Expected** | **+3.5-7.5%** |

**Target Accuracy**: 96-98%+ (up from 95.6%)

## How to Use

1. **Open the notebook**:
   ```
   notebooks/model_improvement.ipynb
   ```

2. **Run all cells**:
   - This will compare multiple models
   - Perform hyperparameter tuning
   - Create ensemble model
   - Save the best model

3. **Check results**:
   - Compare CV scores for each model
   - Final test set accuracy
   - Choose the best performing model

4. **Update app.py** (if using improved model):
   ```python
   # Change model path from:
   model_path = 'models/mental_health_model.pkl'
   # To:
   model_path = 'models/mental_health_model_improved.pkl'
   ```

## Model Files Created

- `models/mental_health_model_improved.pkl` - Best model
- `models/tfidf_vectorizer_improved.pkl` - Enhanced vectorizer
- `models/feature_selector.pkl` - Feature selector (if used)

## Tips for Further Improvement

1. **More Data**: Collect more training data
2. **Data Augmentation**: Use techniques like synonym replacement
3. **Deep Learning**: Try LSTM/Transformer models (BERT, RoBERTa)
4. **Feature Engineering**: Add sentiment scores, emotion detection
5. **Ensemble More Models**: Add more diverse models to ensemble

## Performance Comparison

| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| Logistic Regression (baseline) | 95.6% | Fast (~1 min) |
| XGBoost | 96-97% | Medium (~5 min) |
| Tuned XGBoost | 97-97.5% | Medium (~10 min) |
| Ensemble | 97.5-98%+ | Slow (~15 min) |

## Notes

- **Training time** increases with complexity
- **Ensemble models** are slower for predictions but more accurate
- **XGBoost** usually provides best balance of accuracy and speed
- **Feature selection** can reduce overfitting on small datasets

## Troubleshooting

### If XGBoost not available:
```bash
pip install xgboost
```

### If training is too slow:
- Reduce `n_estimators` in models
- Use fewer CV folds (3 instead of 5)
- Reduce feature count (6000 instead of 8000)

### If accuracy doesn't improve:
- Check data quality
- Try different feature selection methods
- Experiment with different n-gram ranges
- Consider collecting more data

