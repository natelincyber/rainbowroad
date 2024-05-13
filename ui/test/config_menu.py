import tkinter as tk
from tkinter import messagebox

class CircleLightUp:
    def __init__(self, master):
        self.master = master
        self.master.title("Circle Light Up")

        self.canvas_width = 400
        self.canvas_height = 400

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.circle_radius = 30
        self.circle_x = self.canvas_width // 2
        self.circle_y = self.canvas_height // 2
        self.circle = self.canvas.create_oval(self.circle_x - self.circle_radius, self.circle_y - self.circle_radius,
                                              self.circle_x + self.circle_radius, self.circle_y + self.circle_radius, fill="gray")

        # Bind keyboard events
        self.key_bindings = {'a': 'red', 'b': 'green', 'c': 'blue'}  # Default key bindings
        self.bind_keys()

        # Create menu
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)
        self.create_submenu()

    def bind_keys(self):
        self.master.bind('<KeyPress>', self.light_up_circle)
        self.master.bind('<KeyRelease>', self.darken_circle)

    def light_up_circle(self, event):
        key = event.char.lower()
        if key in self.key_bindings:
            self.canvas.itemconfig(self.circle, fill=self.key_bindings[key])

    def darken_circle(self, event=None):
        self.canvas.itemconfig(self.circle, fill="gray")

    def create_submenu(self):
        submenu = tk.Menu(self.menu_bar, tearoff=0)
        submenu.add_command(label="Change Key Bindings", command=self.open_settings_window)
        self.menu_bar.add_cascade(label="Settings", menu=submenu)

    def open_settings_window(self):
        settings_window = tk.Toplevel(self.master)
        settings_window.title("Change Key Bindings")

        instructions_label = tk.Label(settings_window, text="Enter new key bindings (e.g., a:red, b:green, c:blue)")
        instructions_label.pack()

        entry_var = tk.StringVar()
        entry = tk.Entry(settings_window, textvariable=entry_var)
        entry.pack()

        def update_bindings():
            try:
                new_bindings = dict(entry_var.get().split(','))
                self.key_bindings = {key.strip(): value.strip() for key, value in new_bindings.items()}
                self.bind_keys()
                messagebox.showinfo("Success", "Key bindings updated successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        update_button = tk.Button(settings_window, text="Update", command=update_bindings)
        update_button.pack()

def main():
    root = tk.Tk()
    app = CircleLightUp(root)
    root.mainloop()

if __name__ == "__main__":
    main()