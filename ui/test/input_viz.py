import tkinter as tk
import vgamepad as vg
import threading

# POC for input visualization
class InputVisualization:
    def __init__(self, master):

        self.gamepad = vg.VX360Gamepad()

        self.master = master
        self.master.title("Bar Graph")

        self.canvas_width = 400
        self.canvas_height = 300

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.bar_height = 200
        self.bar_width = 50
        self.bar_x = 50
        self.bar_y = self.canvas_height - self.bar_height
        self.bar = self.canvas.create_rectangle(self.bar_x, self.bar_y, self.bar_x + self.bar_width, self.bar_y + self.bar_height, fill="blue")

        self.scale_factor = 2

        self.scale_var = tk.DoubleVar()
        self.scale = tk.Scale(self.master, variable=self.scale_var, from_=0, to=self.canvas_height // self.scale_factor, orient=tk.VERTICAL, command=self.update_bar)
        self.scale.pack()
        self.scale.set(100)

        self.circle_radius = 30
        self.a_x = 225
        self.a_y = 200
        self.a = self.canvas.create_oval(self.a_x - self.circle_radius, self.a_y - self.circle_radius,
                                              self.a_x + self.circle_radius, self.a_y + self.circle_radius, fill="gray")
        
        self.b_x = 275
        self.b_y = 150
        self.b = self.canvas.create_oval(self.b_x - self.circle_radius, self.b_y - self.circle_radius,
                                              self.b_x + self.circle_radius, self.b_y + self.circle_radius, fill="gray")
        
        self.x_x = 175
        self.x_y = 150
        self.x = self.canvas.create_oval(self.x_x - self.circle_radius, self.x_y - self.circle_radius,
                                              self.x_x + self.circle_radius, self.x_y + self.circle_radius, fill="gray")
        
        self.y_x = 225
        self.y_y = 100
        self.y = self.canvas.create_oval(self.y_x - self.circle_radius, self.y_y - self.circle_radius,
                                              self.y_x + self.circle_radius, self.y_y + self.circle_radius, fill="gray")


        # Bind keyboard events
        self.master.bind('<KeyPress-a>', self.light_up_circle)
        self.master.bind('<KeyPress-b>', self.light_up_circle)
        self.master.bind('<KeyPress-c>', self.light_up_circle)
        self.master.bind('<KeyRelease>', self.darken_circle)
        self.darken_circle()

    def gamepadsetup(self):
        self.master.event_generate()

    def light_up_circle(self, event):
        key = event.char.lower()
        if key == 'a':
            self.canvas.itemconfig(self.a, fill="red")
        elif key == 'b':
            self.canvas.itemconfig(self.a, fill="green")
        elif key == 'c':
            self.canvas.itemconfig(self.a, fill="blue")

    def darken_circle(self, event=None):
        self.canvas.itemconfig(self.a, fill="gray")

    def update_bar(self, value):
        new_height = float(value) * self.scale_factor
        self.canvas.coords(self.bar, self.bar_x, self.canvas_height - new_height, self.bar_x + self.bar_width, self.canvas_height)



def test(app):
    import time
    while True: 
        app.gamepad.left_joystick(x_value=-10000, y_value=0)  # values between -32768 and 32767
        app.gamepad.right_trigger_float(value_float=0.5)
        app.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)  # press the A button
        app.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)  # press the left hat button

        app.gamepad.update()  # send the updated state to the computer
        time.sleep(0.5)


        app.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)  # release the A button
        app.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)  # release the A button

        app.gamepad.left_joystick(x_value=0, y_value=0)  # values between -32768 and 32767
        app.gamepad.right_trigger_float(value_float=0)
        app.gamepad.update()  # send the updated state to the computer
        time.sleep(0.5)

def main():

    root = tk.Tk()
    app = InputVisualization(root)
    root.bind()
    # thread = threading.Thread(target=test, args=(app,))
    # thread.start()
    root.mainloop()

if __name__ == "__main__":
    main()