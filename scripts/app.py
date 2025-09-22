import gradio as gr
from gtts import gTTS

# --- Gradio UI Functions ---

def generate_audio_response(text):
    """
    Converts the agent's text response into an audio file using gTTS.
    """
    if not text:
        return None
    output_path = "assistant_response.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)
    return output_path

def respond(text_query, history_state):
    """
    The main interaction function called by the Gradio interface.
    """
    if not text_query or not text_query.strip():
        formatted_history = "\n".join([f"**You:** {turn['user']}\n**Agent:** {turn['assistant']}" for turn in history_state])
        return "", history_state, formatted_history

    query = text_query.strip()
    assistant_response_text = agent.get_rag_response(query, history_state)
    
    new_history = history_state + [{'user': query, 'assistant': assistant_response_text}]
    formatted_history = "\n".join([f"**You:** {turn['user']}\n**Agent:** {turn['assistant']}" for turn in new_history])
    
    return assistant_response_text, new_history, formatted_history

# --- Launch the Gradio Web Interface ---
print("Launching Gradio Interface...")

# Instantiate the agent from agent.py
agent = CustomerServiceAgent()

# Define the UI layout
with gr.Blocks(theme=gr.themes.Soft(), title="Customer Service Agent") as app:
    gr.Markdown("# Advanced Customer Service Agent")
    gr.Markdown("Type your query below and press Submit.")

    history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Your Question", lines=4, placeholder="Type your question here...")
            text_submit_btn = gr.Button("Submit")
            
            with gr.Accordion("Agent's Response", open=True):
                agent_response_text = gr.Textbox(label="Response Text", interactive=False, lines=4)
                with gr.Row():
                    read_aloud_btn = gr.Button("Read Response Aloud")
                    audio_output = gr.Audio(label="Agent's Voice", autoplay=False)

        with gr.Column(scale=3):
            history_display = gr.Markdown("Conversation history will appear here.", label="Conversation")

    text_submit_btn.click(
        fn=respond,
        inputs=[text_input, history_state],
        outputs=[agent_response_text, history_state, history_display]
    ).then(lambda: "", outputs=[text_input])

    read_aloud_btn.click(
        fn=generate_audio_response,
        inputs=[agent_response_text],
        outputs=[audio_output]
    )

app.launch(debug=True, share=True)