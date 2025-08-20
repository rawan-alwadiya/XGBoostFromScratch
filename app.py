import streamlit as st
import numpy as np
import joblib

class SimpleDecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, grad):
        m, n = X.shape
        best_loss = float('inf')
        
        
        for j in range(n):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_val = grad[left_mask].mean()
                right_val = grad[right_mask].mean()

                
                loss = ((grad[left_mask] - left_val)**2).sum() + ((grad[right_mask] - right_val)**2).sum()

                if loss < best_loss:
                    best_loss = loss
                    self.feature_index = j
                    self.threshold = t
                    self.left_value = left_val
                    self.right_value = right_val

    def predict(self, x):
        if x[self.feature_index] <= self.threshold:
            return self.left_value
        else:
            return self.right_value


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class XGBoostScratch:
    def __init__(self, n_rounds=10, lr=0.3):
        self.n_rounds = n_rounds
        self.lr = lr
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        y_onehot = np.eye(K)[y]
        logits = np.zeros((X.shape[0], K))

        self.trees = [[] for _ in range(K)]

        for _ in range(self.n_rounds):
            probs = softmax(logits)

            for k in range(K):
                grad = probs[:, k] - y_onehot[:, k]   
                tree = SimpleDecisionTree(max_depth=1)
                tree.fit(X, grad)
                update = np.array([tree.predict(x) for x in X])

                logits[:, k] -= self.lr * update
                self.trees[k].append(tree)

    def predict(self, X):
        logits = np.zeros((X.shape[0], len(self.classes_)))
        for k, trees in enumerate(self.trees):
            for tree in trees:
                logits[:, k] -= self.lr * np.array([tree.predict(x) for x in X])
        probs = softmax(logits)
        return np.argmax(probs, axis=1)
    

model = joblib.load("XGBoost.pkl")

st.markdown("<h1 style='text-align: center;'>🌙 Moon Mood Classifier</h1>", unsafe_allow_html=True)
st.write("### Predict if the moon is **Happy 🌝** or **Sad 🌚** based on its glow and shadow!")
st.write("Adjust the sliders to see which mood the moon is in!")


x1 = st.slider("🌌 Moon Position", min_value=-1.0, max_value=2.0, value=0.5, step=0.01)
x2 = st.slider("✨ Moon Glow", min_value=-0.5, max_value=1.0, value=0.25, step=0.01)


if st.button("🔮 Predict"):
    features = np.array([[x1, x2]])
    prediction = model.predict(features)[0]

    if prediction == 0:
        st.success("🌝 Happy Moon! The moon is glowing bright and cheerful ✨😀")
    else:
        st.error("🌚 Sad Moon... The moon feels gloomy tonight 😢💤")
