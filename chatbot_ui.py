import tkinter as tk
from tkinter import scrolledtext
from main import get_response  # Ensure you import the get_response function

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

        # Configure the input box for gradient effect
        self.user_input.config(highlightthickness=2, highlightbackground="#5DA3FA", highlightcolor="#00FF00")
        self.user_input.bind("<FocusIn>", self.on_focus_in)
        self.user_input.bind("<FocusOut>", self.on_focus_out)

        self.send_button = tk.Button(self.input_frame, text="Send", font=("Helvetica", 12), command=self.handle_message, bg="#BFBFBF", fg="black", relief=tk.FLAT)
        self.send_button.pack(padx=10, pady=10, side=tk.RIGHT)

    def on_focus_in(self, event):
        self.user_input.config(highlightbackground="#00FF00", highlightcolor="#5DA3FA", highlightthickness=3)

    def on_focus_out(self, event):
        self.user_input.config(highlightbackground="#5DA3FA", highlightcolor="#00FF00", highlightthickness=2)

    def handle_message(self, event=None):
        user_message = self.user_input.get()

        if user_message.strip():
            self.display_message(f"You: {user_message}", align="right")

        response = get_response(user_message)  # Call the response function
        self.display_message(f"Bot: {response}", align="left")

        self.user_input.delete(0, tk.END)

    def display_message(self, message, align="left"):
        self.chat_frame.config(state="normal")

        if align == "right":
            self.chat_frame.tag_configure("user", justify="right", foreground="lightblue")
            self.chat_frame.insert(tk.END, message + "\n", "user")
        else:
            self.chat_frame.tag_configure("bot", justify="left", foreground="#ffffff")
            self.chat_frame.insert(tk.END, message + "\n", "bot")

        self.chat_frame.config(state="disabled")
        self.chat_frame.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotUI(root)
    root.mainloop()
