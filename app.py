# app.py

import gradio as gr
from src.rag_pipeline import rag_query

def answer_question(user_question):
    if not user_question.strip():
        return "", []

    result = rag_query(user_question)
    answer = result["generated_answer"]
    
    sources = [
        f"Source {i+1} (Product: {r.get('product', 'Unknown')}):\n{r['text']}"
        for i, r in enumerate(result["retrieved_contexts"])
    ]
    return answer, sources

# Gradio Interface
with gr.Blocks(title="CrediTrust Complaint Chatbot") as demo:
    gr.Markdown("## 💬 CrediTrust Complaint Insight Assistant")
    gr.Markdown("Ask about customer complaints for Credit Cards, BNPL, Personal Loans, and more.")

    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. Why are users unhappy with Buy Now, Pay Later?",
            lines=1
        )
    
    ask_button = gr.Button("Ask")
    clear_button = gr.Button("Clear")

    answer_output = gr.Textbox(label="🧠 AI-Generated Answer", lines=5)
    sources_output = gr.HighlightedText(label="🔍 Retrieved Sources (Excerpts)", combine_adjacent=True)

    def clear_fields():
        return "", [], gr.update(value="")

    ask_button.click(fn=answer_question, inputs=question_input, outputs=[answer_output, sources_output])
    clear_button.click(fn=clear_fields, outputs=[question_input, answer_output, sources_output])

# Launch
if __name__ == "__main__":
    demo.launch()
