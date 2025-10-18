import gradio as gr
from gtts import gTTS
from agent import CustomerServiceAgent

# --- Gradio UI Functions ---

def generate_audio_response(text):
    """
    Converts the agent's text response into an audio file using gTTS.
    """
    if not text or not text.strip():
        return None
    output_path = "assistant_response.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)
    return output_path

def respond(text_query, history_state):
    """
    The main interaction function. It calls the agent, updates the history state,
    and formats the history for Markdown display.
    """
    if not text_query or not text_query.strip():
        # If input is empty, just return the current state without changes
        formatted_history = "\n---\n".join([f"**You:** {turn['user']}\n\n**Agent:** {turn['assistant']}" for turn in history_state])
        return "", history_state, formatted_history

    # Get the response from the from-scratch agent, passing the query and history
    assistant_response = agent.get_rag_response(text_query, history_state)
    
    # Append the new turn to the history state (list of dictionaries)
    new_history = history_state + [{'user': text_query, 'assistant': assistant_response}]
    
    # Format the entire history for display in the Markdown component
    formatted_history = "\n---\n".join([f"**You:** {turn['user']}\n\n**Agent:** {turn['assistant']}" for turn in new_history])
    
    return assistant_response, new_history, formatted_history

# --- Launch the Gradio Web Interface ---
print("Launching Gradio Interface for IMPROVED From-Scratch Agent...")

# Instantiate the agent from agent_scratch_improved.py
agent = CustomerServiceAgent()

# Define the UI layout using Gradio Blocks for more control
with gr.Blocks(theme=gr.themes.Soft(), title="Advanced Customer Service Agent") as app:
    gr.Markdown("# 🤖 Advanced Customer Service Agent (From Scratch - Improved)")
    gr.Markdown("Type your query in the text box and press the 'Submit' button.")

    # State to hold the conversation history as a list of dictionaries
    history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Your Question",
                lines=4,
                placeholder="e.g., 'My package hasn't arrived yet.'"
            )
            text_submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Accordion("Agent's Latest Response", open=True):
                agent_response_text = gr.Textbox(
                    label="Response Text",
                    interactive=False,
                    lines=4
                )
                with gr.Row():
                    read_aloud_btn = gr.Button("Read Response Aloud")
                    audio_output = gr.Audio(label="Agent's Voice", autoplay=False)
        
        with gr.Column(scale=3):
            history_display = gr.Markdown(
                "Conversation history will appear here.",
                label="Full Conversation"
            )

    # Connect the UI components to the functions
    text_submit_btn.click(
        fn=respond,
        inputs=[text_input, history_state],
        outputs=[agent_response_text, history_state, history_display]
    ).then(lambda: "", outputs=[text_input]) # Clears the input box after submit

    read_aloud_btn.click(
        fn=generate_audio_response,
        inputs=[agent_response_text],
        outputs=[audio_output]
    )
    
    gr.Markdown("--- \n *Project based on the roadmap by Gemini.*")

# Launch the app with a public share link
app.launch(debug=True, share=True)
