# app.py
import gradio as gr
from qa_engine import get_answer

def answer_question(user_question):
    response = get_answer(user_question)
    return response

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Posez votre question"),
    outputs=gr.Textbox(label="Réponse"),
    title="Q&A Chatbot",
    description="Posez une question sur les données économiques"
)

if __name__ == "__main__":
    iface.launch()



