import tkinter as tk
from tkinter import messagebox, Button


# Simple test class for testing tkinter transparent overlays
# works with windows 11
class OverlayWindow:
    def __init__(self, master):
        self.master = master
        self.master.attributes("-alpha", 1)  # Set transparency level
        # Set the root window background color to a transparent color
        self.master.config(bg='')
        self.master.overrideredirect(True)  # Remove window borders
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Calculate position for top right corner
        x_position = screen_width - 300  # Adjust as needed
        y_position = 0

        self.master.geometry(f"300x100+{x_position}+{y_position}")  # Set window size and position
        self.master.lift()  # Lift window to the top
        self.master.wm_attributes("-topmost", True)  # Keep window on top of others


        # Display text
        self.text_label = tk.Label(self.master, text="TEST", font=("Helvetica", 16))
        self.text_label.pack()

        # Place text on top right
        self.text_label.place(relx=1, rely=0, anchor='ne')

        #exit button
        self.exit = Button(self.master, text="exit", command=self.close_window)
        self.exit.pack()
        self.exit.place(relx=0,rely=0, anchor='nw')


    def close_window(self):
        self.master.destroy()

def show_overlay():
    root = tk.Tk()
    overlay = OverlayWindow(root)
    root.mainloop()

show_overlay()
