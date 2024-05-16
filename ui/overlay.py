import tkinter as tk
import vgamepad as vg
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from input.gamepad import GamepadWrapper, Event


# POC for input visualization
class InputVisualization:
    def __init__(self, master):
        
        self.gamepadState = {}
        self.wrapper = GamepadWrapper(self.handle_gamepad)
        self.wrapper.start()

        self.master = master
        self.master.title("OVERLAY")
        self.master.protocol("WM_DELETE_WINDOW", self.onclose)

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.canvas_width = 400
        self.canvas_height = 300
        # Calculate position for top right corner
        x_position = screen_width - self.canvas_width  # Adjust as needed
        y_position = 0

        self.master.geometry(f"{self.canvas_width}x{self.canvas_height}+{x_position}+{y_position}")  # Set window size and position
        self.master.lift()  # Lift window to the top
        self.master.wm_attributes("-topmost", True)  # Keep window on top of others

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, background=self.master["bg"])
        self.canvas.pack()
        
        self.rj_center_x = self.canvas.winfo_reqwidth() // 4
        self.rj_center_y = self.canvas.winfo_reqheight() // 4 + 3
        self.radius= 70
        
        self.canvas.create_oval(
            self.rj_center_x - self.radius, self.rj_center_y - self.radius,
            self.rj_center_x + self.radius, self.rj_center_y + self.radius,
            outline="black", width=2
        )

        self.dot_radius = 5
        self.rj_dot = self.canvas.create_oval(
            self.rj_center_x - self.dot_radius, self.rj_center_y - self.dot_radius,
            self.rj_center_x + self.dot_radius, self.rj_center_y + self.dot_radius,
            fill="red", outline=""
        )


        self.lj_center_x = self.canvas.winfo_reqwidth() // 4
        self.lj_center_y = self.canvas.winfo_reqheight() // 4 + 150
        self.canvas.create_oval(
            self.lj_center_x - self.radius, self.lj_center_y - self.radius,
            self.lj_center_x + self.radius, self.lj_center_y + self.radius,
            outline="black", width=2
        )
        self.lj_dot = self.canvas.create_oval(
            self.lj_center_x - self.dot_radius, self.lj_center_y - self.dot_radius,
            self.lj_center_x + self.dot_radius, self.lj_center_y + self.dot_radius,
            fill="blue", outline=""
        )
    
        self.scale_factor = 2

        self.circle_radius = 30
        self.a_x = 225 + 50
        self.a_y = 200
        self.a = self.canvas.create_oval(self.a_x - self.circle_radius, self.a_y - self.circle_radius,
                                              self.a_x + self.circle_radius, self.a_y + self.circle_radius, fill="gray")
        
        self.b_x = 275 + 50
        self.b_y = 150
        self.b = self.canvas.create_oval(self.b_x - self.circle_radius, self.b_y - self.circle_radius,
                                              self.b_x + self.circle_radius, self.b_y + self.circle_radius, fill="gray")
        
        self.x_x = 175 + 50
        self.x_y = 150
        self.x = self.canvas.create_oval(self.x_x - self.circle_radius, self.x_y - self.circle_radius,
                                              self.x_x + self.circle_radius, self.x_y + self.circle_radius, fill="gray")
        
        self.y_x = 225 + 50
        self.y_y = 100
        self.y = self.canvas.create_oval(self.y_x - self.circle_radius, self.y_y - self.circle_radius,
                                              self.y_x + self.circle_radius, self.y_y + self.circle_radius, fill="gray")

        self.test_on()

        # Bind gamepad events
        self.darken_circle()


    def onclose(self):
        self.wrapper.stop() # MUST STOP THE WRAPPER THREAD OR IT WILL
        self.master.destroy()

    def handle_gamepad(self, event):
        event.run()
        if 'joystick' in event.event.__name__:
            self.gamepadState[event.event.__name__] = [event.info[event.event.__name__][1]["x_value"], event.info[event.event.__name__][1]["y_value"]]
            self.update_joystick(event.event.__name__)
        elif 'button' in event.event.__name__:

            button_value = event.info[event.event.__name__][0][0]
            if 'press' in event.event.__name__:
                
                if button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_A:
                    self.canvas.itemconfig(self.a, fill="red")
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_B:
                    self.canvas.itemconfig(self.b, fill="green")
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_X:
                    self.canvas.itemconfig(self.x, fill="blue")
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_Y:
                    self.canvas.itemconfig(self.y, fill="yellow")

            elif 'release' in event.event.__name__:
                if button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_A:
                    self.canvas.itemconfig(self.a, fill="gray")
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_B:
                    self.canvas.itemconfig(self.b, fill="gray")
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_X:
                    self.canvas.itemconfig(self.x, fill="gray")
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_Y:
                    self.canvas.itemconfig(self.y, fill="gray")
        self.wrapper.gamepad.update()

    def update_joystick(self, stick):
            max_distance = self.radius - self.dot_radius
            x, y = self.gamepadState[stick]
            distance = (x**2 + y**2)**0.5
            if distance > max_distance:
                    x = x * max_distance / distance
                    y = y * max_distance / distance
            if stick == "right_joystick":
                self.canvas.coords(self.rj_dot, self.rj_center_x + x - self.dot_radius, self.rj_center_y + y - self.dot_radius,
                            self.rj_center_x + x + self.dot_radius, self.rj_center_y + y + self.dot_radius)
            else:
                self.canvas.coords(self.lj_dot, self.lj_center_x + x - self.dot_radius, self.lj_center_y + y - self.dot_radius,
                            self.lj_center_x + x + self.dot_radius, self.lj_center_y + y + self.dot_radius)
            

    def darken_circle(self, event=None):
        self.canvas.itemconfig(self.a, fill="gray")
    
    def test_on(self):
        ev = Event(self.wrapper.gamepad.right_joystick, x_value=50, y_value=0)
        ev2 = Event(self.wrapper.gamepad.left_joystick, x_value=0, y_value=50)
        ev3 = Event(self.wrapper.gamepad.press_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        self.wrapper.addEvent(ev)
        self.wrapper.addEvent(ev2)
        self.wrapper.addEvent(ev3)
        self.master.after(500, self.test_off)


    def test_off(self):
        ev = Event(self.wrapper.gamepad.right_joystick,  x_value=0, y_value=0)
        ev2 = Event(self.wrapper.gamepad.left_joystick, x_value=0, y_value=0)
        ev3 = Event(self.wrapper.gamepad.release_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        self.wrapper.addEvent(ev)
        self.wrapper.addEvent(ev2)
        self.wrapper.addEvent(ev3)
        self.master.after(500, self.test_on)

def main():

    root = tk.Tk()
    app = InputVisualization(root)
    root.mainloop()

if __name__ == "__main__":
    main()