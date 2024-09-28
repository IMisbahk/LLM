import tkinter as tk
from tkinter import scrolledtext
import torch
from models.transformer_model import TransformerModel
from utils.tokenizer import SimpleTokenizer

class ChatbotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        self.root.geometry("800x600")
        self.root.configure(bg="#121212") 

        # Initialize tokenizer and model
        self.vocab = ['<pad>', '<unk>', 'hello', 'world', 'this', 'is', 'a', 'test']
        self.tokenizer = SimpleTokenizer(self.vocab)
        self.model = TransformerModel(len(self.vocab))

        self.chat_frame = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state="disabled", padx=10, pady=10)
        self.chat_frame.config(bg="#1E1E1E", font=("Helvetica", 12), fg="white")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Input frame
        self.input_frame = tk.Frame(self.root, bg="#121212")
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        self.user_input = tk.Entry(self.input_frame, font=("Helvetica", 14), bg="#000000", fg="white", bd=0, insertbackground="white")
        self.user_input.pack(fill=tk.X, padx=10, pady=10, ipady=8, side=tk.LEFT, expand=True)
        self.user_input.bind("<Return>", self.handle_message)

        self.user_input.config(highlightthickness=2)
        self.user_input.config(highlightbackground="#5DA3FA", highlightcolor="#00FF00")

        self.send_button = tk.Button(self.input_frame, text="Send", font=("Helvetica", 12), command=self.handle_message, bg="#BFBFBF", fg="black", relief=tk.FLAT)
        self.send_button.pack(padx=10, pady=10, side=tk.RIGHT)

    def handle_message(self, event=None):
        user_message = self.user_input.get()

        if user_message.strip():
            self.display_message(f"You: {user_message}", align="right")

            # Get response from the model
            response = self.get_response(user_message)
            self.display_message(f"Bot: {response}", align="left")

            self.user_input.delete(0, tk.END)

    def get_response(self, message):
        # Tokenize input message
        input_tokens = self.tokenizer.tokenize(message).unsqueeze(0)  # Add batch dimension

        # Disable gradient calculation for inference
        with torch.no_grad():
            output = self.model(input_tokens)

        # Get predicted tokens and detokenize
        predicted_tokens = output.argmax(dim=-1).squeeze(0)  # Get the highest probability tokens
        output_text = self.tokenizer.detokenize(predicted_tokens.tolist())

        return output_text

    def display_message(self, message, align="left"):
        self.chat_frame.config(state="normal")

        if align == "right":
            self.chat_frame.tag_configure("user", justify="right", foreground="#B3E5FC")
            self.chat_frame.insert(tk.END, message + "\n", "user")
        else:
            self.chat_frame.tag_configure("bot", justify="left", foreground="#EEEEEE")
            self.chat_frame.insert(tk.END, message + "\n", "bot")

        self.chat_frame.config(state="disabled")
        self.chat_frame.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotUI(root)
    root.mainloop()
