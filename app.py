 # 💻 Interface web Streamlit
import streamlit as st
from utils import load_dataset, answer_question

st.set_page_config(page_title="Public Data Q&A Bot", layout="wide")

st.title("📊 Public Data Q&A Chatbot")
st.write("Ask a question about public datasets and get a data-driven answer!")

# Zone de saisie de la question
user_question = st.text_input("Ask your question:", "")

# Quand l'utilisateur clique sur le bouton
if st.button("Submit") and user_question:
    with st.spinner("Analyzing data and generating answer..."):
        # Charger le dataset (à adapter selon ton projet)
        df = load_dataset("data/sample.csv")  # ← fichier de test à créer ou modifier
        # Générer une réponse
        answer, fig = answer_question(user_question, df)
        st.markdown("### 🧠 Answer")
        st.write(answer)
        # Affichage du graphique
        if fig:
            st.markdown("### 📈 Visualization")
            st.pyplot(fig)
