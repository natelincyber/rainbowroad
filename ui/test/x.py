import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QBrush, QColor
import vgamepad as vg
import sys, os, time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..'))
sys.path.append(parent_dir)

from input.gamepad import GamepadWrapper, Event
class InputVisualization(QWidget):
    def __init__(self):
        super().__init__()
        
        self.gamepadState = {}
        self.wrapper = GamepadWrapper(self.handle_gamepad)
        self.wrapper.start()

        self.setWindowTitle("OVERLAY")
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        screen_width = QApplication.desktop().screenGeometry().width()
        screen_height = QApplication.desktop().screenGeometry().height()
        self.canvas_width = 400
        self.canvas_height = 300
        # Calculate position for top right corner
        x_position = screen_width - self.canvas_width  # Adjust as needed
        y_position = 0

        self.setGeometry(x_position, y_position, self.canvas_width, self.canvas_height)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, self.canvas_width, self.canvas_height)

        self.rj_center_x = self.canvas_width // 4
        self.rj_center_y = self.canvas_height // 4 + 3
        self.radius= 70
        self.scene.addEllipse(self.rj_center_x - self.radius, self.rj_center_y - self.radius,
                              self.radius * 2, self.radius * 2)
        self.dot_radius = 5
        self.rj_dot = self.scene.addEllipse(self.rj_center_x - self.dot_radius, self.rj_center_y - self.dot_radius,
                                             self.dot_radius * 2, self.dot_radius * 2, Qt.black, Qt.red)

        self.lj_center_x = self.canvas_width // 4
        self.lj_center_y = self.canvas_height // 4 + 150
        self.scene.addEllipse(self.lj_center_x - self.radius, self.lj_center_y - self.radius,
                              self.radius * 2, self.radius * 2)
        self.lj_dot = self.scene.addEllipse(self.lj_center_x - self.dot_radius, self.lj_center_y - self.dot_radius,
                                             self.dot_radius * 2, self.dot_radius * 2)
        
        self.circle_radius = 30
        self.a_x = 225 + 50
        self.a_y = 200
        self.a = self.scene.addEllipse(self.a_x - self.circle_radius, self.a_y - self.circle_radius,
                                        self.circle_radius * 2, self.circle_radius * 2)

        self.b_x = 275 + 50
        self.b_y = 150
        self.b = self.scene.addEllipse(self.b_x - self.circle_radius, self.b_y - self.circle_radius,
                                        self.circle_radius * 2, self.circle_radius * 2)

        self.x_x = 175 + 50
        self.x_y = 150
        self.x = self.scene.addEllipse(self.x_x - self.circle_radius, self.x_y - self.circle_radius,
                                        self.circle_radius * 2, self.circle_radius * 2)

        self.y_x = 225 + 50
        self.y_y = 100
        self.y = self.scene.addEllipse(self.y_x - self.circle_radius, self.y_y - self.circle_radius,
                                        self.circle_radius * 2, self.circle_radius * 2)
        self.view.show()
        self.test_on()

        # Make the window transparent
        # self.setWindowOpacity(0.7)

    def handle_gamepad(self, event):
        event.run()
        if 'joystick' in event.event.__name__:
            self.gamepadState[event.event.__name__] = [event.info[event.event.__name__][1]["x_value"], event.info[event.event.__name__][1]["y_value"]]
            self.update_joystick(event.event.__name__)
        elif 'button' in event.event.__name__:

            button_value = event.info[event.event.__name__][0][0]
            if 'press' in event.event.__name__:
                
                if button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_A:
                    self.a.setBrush(Qt.red)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_B:
                    self.b.setBrush(Qt.green)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_X:
                    self.x.setBrush(Qt.blue)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_Y:
                    self.y.setBrush(Qt.yellow)

            elif 'release' in event.event.__name__:
                if button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_A:
                    self.a.setBrush(Qt.gray)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_B:
                    self.b.setBrush(Qt.gray)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_X:
                    self.x.setBrush(Qt.gray)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_Y:
                    self.y.setBrush(Qt.gray)
        self.wrapper.gamepad.update()

    def update_joystick(self, stick):
        max_distance = self.radius - self.dot_radius
        x, y = self.gamepadState[stick]
        distance = (x**2 + y**2)**0.5
        if distance > max_distance:
            x = x * max_distance / distance
            y = y * max_distance / distance
        if stick == "right_joystick":
            self.rj_dot.setRect(self.rj_center_x + x - self.dot_radius, self.rj_center_y + y - self.dot_radius,
                                 self.dot_radius * 2, self.dot_radius * 2)
        else:
            self.lj_dot.setRect(self.lj_center_x + x - self.dot_radius, self.lj_center_y + y - self.dot_radius,
                                 self.dot_radius * 2, self.dot_radius * 2)

    def test_on(self):
        ev = Event(self.wrapper.gamepad.right_joystick, x_value=50, y_value=0)
        ev2 = Event(self.wrapper.gamepad.left_joystick, x_value=0, y_value=50)
        ev3 = Event(self.wrapper.gamepad.press_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        self.wrapper.addEvent(ev)
        self.wrapper.addEvent(ev2)
        self.wrapper.addEvent(ev3)
        time.sleep(0.5)
        self.test_off()

    def test_off(self):
        ev = Event(self.wrapper.gamepad.right_joystick,  x_value=0, y_value=0)
        ev2 = Event(self.wrapper.gamepad.left_joystick, x_value=0, y_value=0)
        ev3 = Event(self.wrapper.gamepad.release_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        self.wrapper.addEvent(ev)
        self.wrapper.addEvent(ev2)
        self.wrapper.addEvent(ev3)
        time.sleep(0.5)
        self.test_on()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InputVisualization()
    window.show()
    sys.exit(app.exec_())
