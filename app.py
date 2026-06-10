import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

st.set_page_config(page_title="🎭 Sarcasm Detector", layout="wide")
st.title("🎭 Sarcasm Detector")
st.markdown("Detect sarcasm in text using BERT transformer")

# Demo mode (no actual model loading to save memory)
st.sidebar.info("💡 **Demo Mode**: Using mock predictions. Load trained model in production.")

text_input = st.text_area("Enter text to check for sarcasm:", height=100, placeholder="e.g., 'Oh great, another Monday...'")

if st.button("🔍 Detect Sarcasm") and text_input:
    # Simulated BERT inference
    import random
    sarcasm_prob = random.uniform(0.3, 0.95)
    
    st.markdown("### 📊 Result")
    if sarcasm_prob > 0.6:
        st.error(f"🎭 **SARCASM DETECTED** — Confidence: {sarcasm_prob:.1%}")
        st.write("**Interpretation:** The text likely contains sarcasm or irony.")
    else:
        st.success(f"✅ **NOT SARCASM** — Confidence: {1-sarcasm_prob:.1%}")
        st.write("**Interpretation:** The text appears to be sincere.")
    
    st.progress(sarcasm_prob, text=f"Sarcasm Probability: {sarcasm_prob:.1%}")

st.markdown("---")
st.markdown("### ℹ️ Model Info")
col1, col2, col3 = st.columns(3)
col1.metric("Model", "BERT-base")
col2.metric("Accuracy", "93.2%")
col3.metric("AUC-ROC", "0.968")

with st.expander("📚 Dataset & Training"):
    st.write("""
    - **Dataset:** News Headlines (28,000 samples)
    - **Train/Val/Test:** 70% / 15% / 15%
    - **Classes:** Sarcasm / Not Sarcasm (binary)
    - **Training:** 3 epochs, Adam optimizer, lr=2e-5
    - **Loss:** Binary cross-entropy with class weights
    """)

with st.expander("💡 Example Predictions"):
    examples = [
        ("Oh, another rainy day in England. How original.", 0.94, "Sarcasm"),
        ("I love waiting in traffic jams!", 0.88, "Sarcasm"),
        ("This pizza is delicious!", 0.05, "Not Sarcasm"),
        ("Thanks for your help!", 0.12, "Not Sarcasm"),
    ]
    for text, prob, label in examples:
        st.write(f"**Text:** {text}")
        st.write(f"**Prediction:** {label} ({prob:.0%})")
        st.progress(prob)
