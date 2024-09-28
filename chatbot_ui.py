# chatbot_ui.py
import tkinter as tk
from tkinter import scrolledtext
from main import model, tokenizer, train_model
import torch

class ChatbotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        self.root.geometry("800x600")
        self.root.configure(bg="#121212") 

        self.chat_frame = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state="disabled", padx=10, pady=10)
        self.chat_frame.config(bg="#1E1E1E", font=("Helvetica", 12), fg="white")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Input frame
        self.input_frame = tk.Frame(self.root, bg="#121212")
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        self.user_input = tk.Entry(self.input_frame, font=("Helvetica", 14), bg="#000000", fg="white", bd=0, insertbackground="white")
        self.user_input.pack(fill=tk.X, padx=10, pady=10, ipady=8, side=tk.LEFT, expand=True)
        self.user_input.bind("<Return>", self.handle_message) 

        self.send_button = tk.Button(self.input_frame, text="Send", font=("Helvetica", 12), command=self.handle_message, bg="#BFBFBF", fg="black", relief=tk.FLAT)
        self.send_button.pack(padx=10, pady=10, side=tk.RIGHT)

    def handle_message(self, event=None):
        user_message = self.user_input.get()

        if user_message.strip():
            self.display_message(f"You: {user_message}", align="right")

            # Tokenize input
            input_tokens = tokenizer.tokenize(user_message).unsqueeze(0)  # Add batch dimension
            target_tokens = input_tokens  # For simplicity, using the same input as target for training

            # Train the model with the user's input
            loss = train_model(model, input_tokens, target_tokens, optimizer, criterion)
            response = "This is a placeholder response."  # Replace this with actual response logic

            self.display_message(f"Bot: {response}", align="left")
            self.user_input.delete(0, tk.END)

    def display_message(self, message, align="left"):
        self.chat_frame.config(state="normal")
        self.chat_frame.insert(tk.END, message + "\n", align)
        self.chat_frame.config(state="disabled")
        self.chat_frame.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotUI(root)
    root.mainloop()
