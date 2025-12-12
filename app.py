import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Mental Health Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .depression-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .no-depression {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize NLTK resources
@st.cache_resource
def load_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass
    return True

load_nltk_resources()

# Initialize preprocessing components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess text for model prediction"""
    if pd.isna(text) or text is None:
        return ''
    
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w.isalpha()]
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the improved ensemble model, vectorizer, and feature selector"""
    # Try multiple possible base directories
    possible_base_dirs = [
        'D:/mental_health_detector',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Parent of app.py location
        os.getcwd(),  # Current working directory
    ]
    
    model = None
    vectorizer = None
    selector = None
    
    for base_dir in possible_base_dirs:
        # Try to load improved model first (preferred - higher accuracy)
        improved_model_path = os.path.join(base_dir, 'models/mental_health_model_improved.pkl')
        improved_vectorizer_path = os.path.join(base_dir, 'models/tfidf_vectorizer_improved.pkl')
        selector_path = os.path.join(base_dir, 'models/feature_selector.pkl')
        
        # Fallback to original model
        original_model_path = os.path.join(base_dir, 'models/mental_health_model.pkl')
        original_vectorizer_path = os.path.join(base_dir, 'models/tfidf_vectorizer.pkl')
        
        # Try improved model first
        if os.path.exists(improved_model_path) and os.path.exists(improved_vectorizer_path):
            try:
                with open(improved_model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(improved_vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                # Try to load feature selector if it exists
                if os.path.exists(selector_path):
                    with open(selector_path, 'rb') as f:
                        selector = pickle.load(f)
                    st.sidebar.success(f"✅ Improved ensemble model loaded from: {base_dir}")
                else:
                    st.sidebar.success(f"✅ Improved model loaded from: {base_dir}")
                return model, vectorizer, selector
            except Exception as e:
                st.sidebar.warning(f"Error loading improved model from {base_dir}: {e}")
                continue
        
        # Fallback to original model
        elif os.path.exists(original_model_path) and os.path.exists(original_vectorizer_path):
            try:
                with open(original_model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(original_vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                st.sidebar.info(f"ℹ️ Original model loaded from: {base_dir} (improved model not found)")
                return model, vectorizer, None
            except Exception as e:
                st.sidebar.warning(f"Error loading original model from {base_dir}: {e}")
                continue
    
    # If we get here, model wasn't found
    st.error("""
    ❌ **Model files not found!**
    
    Please ensure:
    1. The improved model has been trained by running `notebooks/model_improvement.ipynb` (preferred)
       OR the original model by running `notebooks/mental_health_detector.ipynb`
    2. Model files exist in the `models/` directory:
       - `models/mental_health_model_improved.pkl` (preferred - ~97-98% accuracy)
       - `models/tfidf_vectorizer_improved.pkl` (preferred)
       - `models/feature_selector.pkl` (optional, for improved model)
       OR
       - `models/mental_health_model.pkl` (fallback - ~95% accuracy)
       - `models/tfidf_vectorizer.pkl` (fallback)
    
    Checked directories:
    """ + "\n".join([f"- {d}" for d in possible_base_dirs]))
    st.stop()
    return None, None, None

model, vectorizer, selector = load_model()

def predict_mental_health(text, depression_threshold=0.40):
    """Predict mental health status from text"""
    if not text or text.strip() == '':
        return {
            'text': text,
            'prediction': -1,
            'label': 'Insufficient text for analysis',
            'confidence': 0.0,
            'depression_probability': 0.0,
            'probabilities': {'No depression': 50.0, 'Depression': 50.0}
        }
    
    processed_text = preprocess_text(text)
    
    if not processed_text or processed_text.strip() == '':
        return {
            'text': text,
            'prediction': -1,
            'label': 'Insufficient text for analysis',
            'confidence': 0.0,
            'depression_probability': 0.0,
            'probabilities': {'No depression': 50.0, 'Depression': 50.0}
        }
    
    # Transform text to TF-IDF features
    text_tfidf = vectorizer.transform([processed_text])
    
    # Apply feature selector if available (for improved model)
    if selector is not None:
        text_tfidf = selector.transform(text_tfidf)
    
    # Get prediction probabilities
    probability = model.predict_proba(text_tfidf)[0]
    no_depression_prob = probability[0]
    depression_prob = probability[1]
    
    # Determine prediction based on which probability is higher
    # But also consider the threshold for sensitivity
    if depression_prob >= depression_threshold and depression_prob > no_depression_prob:
        # Depression is above threshold AND higher than no depression
        prediction = 1
        label = "Depression detected"
        confidence = depression_prob * 100
    elif no_depression_prob > depression_prob:
        # No depression has higher probability
        prediction = 0
        label = "No depression detected"
        confidence = no_depression_prob * 100
    else:
        # Depression is above threshold but lower than no depression
        # In this case, go with the higher probability (no depression)
        prediction = 0
        label = "No depression detected"
        confidence = no_depression_prob * 100
    
    is_borderline = abs(depression_prob - 0.5) < 0.15
    
    return {
        'text': text,
        'prediction': int(prediction),
        'label': label,
        'confidence': round(confidence, 2),
        'depression_probability': round(depression_prob * 100, 2),
        'is_borderline': is_borderline,
        'probabilities': {
            'No depression': round(probability[0] * 100, 2),
            'Depression': round(probability[1] * 100, 2)
        }
    }

def predict_batch(texts, depression_threshold=0.40):
    """Predict mental health status for multiple texts"""
    results = []
    for text in texts:
        result = predict_mental_health(text, depression_threshold)
        results.append(result)
    return results

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">Mental Health Detection Dashboard</h1>', unsafe_allow_html=True)
    if selector is not None:
        st.markdown('<p class="sub-header">✨ Using Improved Ensemble Model (~97-98% accuracy) | Analyze tweets and text for mental health indicators</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="sub-header">Analyze tweets and text for mental health indicators</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        depression_threshold = st.slider(
            "Depression Detection Threshold",
            min_value=0.30,
            max_value=0.50,
            value=0.40,
            step=0.05,
            help="Lower threshold = more sensitive (catches more cases, may have false positives). Higher threshold = more conservative."
        )
        st.info(f"Current threshold: {depression_threshold*100:.0f}%")
        
        st.markdown("---")
        st.header("ℹ️ About")
        
        # Show which model is being used
        if selector is not None:
            model_type = "Improved Ensemble Model"
            model_accuracy = "~97-98%"
            model_info = "Using advanced ensemble model with feature selection for higher accuracy"
        else:
            model_type = "Standard Model"
            model_accuracy = "~95.6%"
            model_info = "Using standard Logistic Regression model"
        
        st.markdown(f"""
        **Model Type**: {model_type}
        
        **Model Accuracy**: {model_accuracy}
        
        {model_info}
        
        This dashboard uses a machine learning model trained on Reddit posts to detect potential mental health issues in text.
        
        **Note**: This tool is for informational purposes only and should not replace professional mental health advice.
        """)
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.metric("Detection Threshold", f"{depression_threshold*100:.0f}%")
        if selector is not None:
            st.success("✨ Using improved ensemble model")
        else:
            st.info("Using standard model")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([" Single Text Analysis", " Batch Upload", "Analytics"])
    
    # Tab 1: Single Text Analysis
    with tab1:
        st.header("Analyze Single Text/Tweet")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter text or tweet to analyze:",
                height=150,
                placeholder="Type or paste your text here...",
                help="Enter any text, tweet, or social media post to analyze for mental health indicators."
            )
        
        with col2:
            st.markdown("### Quick Examples")
            example_texts = {
                "Sad Example": "I'm feeling really down today. Can't stop thinking about negative things. Life feels meaningless.",
                "Happy Example": "Had a great day today! Went for a walk and met some friends. Feeling happy and energized!",
                "Motivation Issue": "I can't find motivation to do anything. Everything feels hopeless.",
                "Suicidal Thoughts": "I've been having suicidal thoughts lately. I don't know what to do anymore."
            }
            
            for label, text in example_texts.items():
                if st.button(f" {label}", key=f"example_{label}"):
                    st.session_state.example_text = text
        
        # Use example text if selected
        if 'example_text' in st.session_state:
            text_input = st.session_state.example_text
            del st.session_state.example_text
        
        if st.button(" Analyze Text", type="primary", use_container_width=True):
            if text_input and text_input.strip():
                with st.spinner("Analyzing text..."):
                    result = predict_mental_health(text_input, depression_threshold)
                
                # Display results
                st.markdown("---")
                st.header(" Analysis Results")
                
                # Prediction box
                if result['prediction'] == 1:
                    st.markdown(f"""
                    <div class="prediction-box depression-detected">
                        <h2> {result['label']}</h2>
                        <p><strong>Depression Probability:</strong> {result['depression_probability']}%</p>
                        <p><strong>Confidence:</strong> {result['confidence']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif result['prediction'] == 0:
                    st.markdown(f"""
                    <div class="prediction-box no-depression">
                        <h2> {result['label']}</h2>
                        <p><strong>No Depression Probability:</strong> {result['probabilities']['No depression']}%</p>
                        <p><strong>Confidence:</strong> {result['confidence']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Insufficient text for analysis. Please provide more text.")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Depression Probability", f"{result['depression_probability']}%")
                with col2:
                    st.metric("No Depression Probability", f"{result['probabilities']['No depression']}%")
                with col3:
                    st.metric("Confidence", f"{result['confidence']}%")
                
                # Probability visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=['No Depression', 'Depression'],
                        y=[result['probabilities']['No depression'], result['probabilities']['Depression']],
                        marker_color=['#4caf50', '#f44336'],
                        text=[f"{result['probabilities']['No depression']}%", f"{result['probabilities']['Depression']}%"],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 100],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Borderline warning
                if result.get('is_borderline', False):
                    st.warning("⚠️ **Borderline Case**: The model is uncertain about this prediction. Results should be interpreted with caution.")
                
                # Disclaimer
                st.info(" **Disclaimer**: This is an automated analysis tool. It should not replace professional mental health evaluation. If you're experiencing mental health issues, please consult with a healthcare professional.")
            else:
                st.warning("Please enter some text to analyze.")
    
    # Tab 2: Batch Upload
    with tab2:
        st.header("Batch Analysis")
        
        upload_option = st.radio(
            "Choose input method:",
            ["Upload CSV file", "Enter multiple texts"],
            horizontal=True
        )
        
        if upload_option == "Upload CSV file":
            uploaded_file = st.file_uploader(
                "Upload a CSV file with a 'text' or 'tweet' column",
                type=['csv'],
                help="CSV file should contain a column named 'text' or 'tweet' with the texts to analyze."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
                    
                    # Find text column
                    text_column = None
                    for col in df.columns:
                        if col.lower() in ['text', 'tweet', 'message', 'post', 'content']:
                            text_column = col
                            break
                    
                    if text_column is None:
                        st.error("❌ Could not find a text column. Please ensure your CSV has a column named 'text', 'tweet', 'message', 'post', or 'content'.")
                        st.dataframe(df.head())
                    else:
                        st.info(f" Using column: **{text_column}**")
                        
                        if st.button(" Analyze All Texts", type="primary", use_container_width=True):
                            texts = df[text_column].dropna().tolist()
                            
                            with st.spinner(f"Analyzing {len(texts)} texts..."):
                                results = predict_batch(texts, depression_threshold)
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Add to original dataframe
                            df['prediction'] = results_df['prediction']
                            df['label'] = results_df['label']
                            df['depression_probability'] = results_df['depression_probability']
                            df['confidence'] = results_df['confidence']
                            
                            # Display results
                            st.markdown("---")
                            st.header(" Batch Analysis Results")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Texts", len(results))
                            with col2:
                                depression_count = sum(1 for r in results if r['prediction'] == 1)
                                st.metric("Depression Detected", depression_count, delta=f"{depression_count/len(results)*100:.1f}%")
                            with col3:
                                avg_depression_prob = np.mean([r['depression_probability'] for r in results])
                                st.metric("Avg Depression Prob", f"{avg_depression_prob:.1f}%")
                            with col4:
                                avg_confidence = np.mean([r['confidence'] for r in results])
                                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                            
                            # Visualization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Distribution pie chart
                                labels = ['No Depression', 'Depression Detected']
                                values = [
                                    sum(1 for r in results if r['prediction'] == 0),
                                    sum(1 for r in results if r['prediction'] == 1)
                                ]
                                fig_pie = px.pie(
                                    values=values,
                                    names=labels,
                                    title="Prediction Distribution",
                                    color_discrete_map={'No Depression': '#4caf50', 'Depression Detected': '#f44336'}
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # Depression probability distribution
                                depression_probs = [r['depression_probability'] for r in results]
                                fig_hist = px.histogram(
                                    x=depression_probs,
                                    nbins=20,
                                    title="Depression Probability Distribution",
                                    labels={'x': 'Depression Probability (%)', 'y': 'Count'},
                                    color_discrete_sequence=['#f44336']
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Results table
                            st.markdown("### 📋 Detailed Results")
                            display_df = df[[text_column, 'label', 'depression_probability', 'confidence']].copy()
                            st.dataframe(display_df, use_container_width=True, height=400)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"mental_health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Enter multiple texts
            st.markdown("Enter multiple texts (one per line):")
            batch_texts_input = st.text_area(
                "Enter texts:",
                height=300,
                placeholder="Enter one text per line...\n\nExample:\nI'm feeling really down today.\nHad a great day!\nI can't find motivation.",
                help="Enter multiple texts, one per line."
            )
            
            if st.button("🔍 Analyze All", type="primary", use_container_width=True):
                if batch_texts_input:
                    texts = [line.strip() for line in batch_texts_input.split('\n') if line.strip()]
                    
                    if texts:
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            results = predict_batch(texts, depression_threshold)
                        
                        # Display results
                        st.markdown("---")
                        st.header("📊 Batch Analysis Results")
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Texts", len(results))
                        with col2:
                            depression_count = sum(1 for r in results if r['prediction'] == 1)
                            st.metric("Depression Detected", depression_count)
                        with col3:
                            avg_depression_prob = np.mean([r['depression_probability'] for r in results])
                            st.metric("Avg Depression Prob", f"{avg_depression_prob:.1f}%")
                        
                        # Results display
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Text {i}: {result['text'][:50]}..."):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Label:** {result['label']}")
                                    st.write(f"**Depression Probability:** {result['depression_probability']}%")
                                with col2:
                                    st.write(f"**Confidence:** {result['confidence']}%")
                                    if result.get('is_borderline', False):
                                        st.warning(" Borderline case")
                                
                                # Mini chart
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=['No Depression', 'Depression'],
                                        y=[result['probabilities']['No depression'], result['probabilities']['Depression']],
                                        marker_color=['#4caf50', '#f44336']
                                    )
                                ])
                                fig.update_layout(height=200, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please enter at least one text.")
                else:
                    st.warning("Please enter some texts to analyze.")
    
    # Tab 3: Analytics
    with tab3:
        st.header(" Analytics & Insights")
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if st.session_state.analysis_history:
            st.subheader("Analysis History")
            
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", len(history_df))
            with col2:
                depression_count = sum(history_df['prediction'] == 1)
                st.metric("Depression Cases", depression_count)
            with col3:
                avg_prob = history_df['depression_probability'].mean()
                st.metric("Avg Depression Prob", f"{avg_prob:.1f}%")
            with col4:
                avg_conf = history_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Trend over time
                if 'timestamp' in history_df.columns:
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    fig_trend = px.line(
                        history_df,
                        x='timestamp',
                        y='depression_probability',
                        title="Depression Probability Trend",
                        labels={'depression_probability': 'Depression Probability (%)', 'timestamp': 'Time'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Distribution
                fig_dist = px.histogram(
                    history_df,
                    x='depression_probability',
                    nbins=20,
                    title="Depression Probability Distribution",
                    labels={'depression_probability': 'Depression Probability (%)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # History table
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.analysis_history = []
                st.rerun()
        else:
            st.info("No analysis history yet. Start analyzing texts to see analytics here.")
            st.markdown("""
            ###  Analytics Features:
            - Track analysis history
            - View trends over time
            - Monitor depression probability distributions
            - Export analytics data
            """)

if __name__ == "__main__":
    main()

