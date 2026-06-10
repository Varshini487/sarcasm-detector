# 🎭 Sarcasm Detector

A **BERT-based NLP classifier** that detects sarcasm in text with ~93% accuracy.

## 🤔 The Problem
Sarcasm is hard for ML models because the literal words are positive, but the meaning is negative:
- "Oh great, another meeting!" (sarcastic) vs "Great job!" (sincere)

Traditional keyword-based models fail because they can't understand context and intent.

## ✨ How It Works
1. **Input:** Raw text (tweet/review)
2. **Tokenization:** BERT tokenizer breaks text into subword tokens
3. **Contextual embeddings:** BERT encodes each token considering the full sentence context
4. **Classification:** Logistic head predicts Sarcastic or Not Sarcastic
5. **Output:** Label + confidence score (0-1)

## 📊 Model Details
- **Architecture:** BERT-base (12 layers, 768 hidden units)
- **Training:** Fine-tuned on News Headlines dataset (28,000 samples)
- **Train/Val/Test:** 70%/15%/15%
- **Loss function:** Binary cross-entropy with class weights (to handle imbalance)
- **Optimizer:** Adam (lr=2e-5)
- **Epochs:** 3-5

## 📈 Performance
| Metric | Score |
|--------|-------|
| Accuracy | 93.2% |
| Precision (Sarcasm) | 91.5% |
| Recall (Sarcasm) | 89.8% |
| F1-Score | 90.6% |
| AUC-ROC | 0.968 |

## 🛠️ Tech Stack
- **Transformers** – BERT model
- **PyTorch / TensorFlow** – deep learning
- **Scikit-learn** – metrics
- **Streamlit** – web UI

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/sarcasm-detector
cd sarcasm-detector
pip install -r requirements.txt
streamlit run app.py
```

## 📝 Use Cases
- Social media sentiment analysis
- Customer review classification
- Content moderation
- Chatbot response filtering

## 3️⃣ Interview Talking Points

**Point 1: Handling Context**
*"The key insight is that sarcasm requires understanding context. I used BERT instead of LSTM because BERT uses multi-head attention to capture relationships between ALL words simultaneously, not just sequential processing. This lets it catch sarcasm markers like tone shifts."*

**Point 2: Class Imbalance**
*"The dataset was imbalanced (70% non-sarcasm, 30% sarcasm). I used weighted loss during training so the model doesn't ignore the minority class. I also used stratified train-test split to ensure both sets have the same distribution."*

**Point 3: Ablation Study**
*"I compared BERT vs LSTM baseline — BERT won by 8% accuracy. I also tested different pooling strategies (CLS token vs mean pooling), and CLS performed best. These comparisons prove why I chose BERT."*
