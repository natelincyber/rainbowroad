from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter
import vgamepad as vg
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from input.gamepad import GamepadWrapper, Event


class CircleWidget(QWidget):
    def __init__(self, padding, radius):
        super().__init__()
        self.padding = padding
        self.circle_radius = radius

        self.setGeometry(0, 0, 300, 300)

        self.brush_top = QBrush(Qt.yellow)
        self.brush_left = QBrush(Qt.blue)
        self.brush_right = QBrush(Qt.red)
        self.brush_bottom = QBrush(Qt.green)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

     
        # Draw circles
        painter.setBrush(self.brush_top)
        painter.drawEllipse((self.width() - self.circle_radius) // 2, self.padding, self.circle_radius, self.circle_radius)

        painter.setBrush(self.brush_left)
        painter.drawEllipse(self.padding, (self.height() - self.circle_radius) // 2, self.circle_radius, self.circle_radius)

        painter.setBrush(self.brush_right)
        painter.drawEllipse(self.width() - self.padding - self.circle_radius, (self.height() - self.circle_radius) // 2, self.circle_radius, self.circle_radius)

        painter.setBrush(self.brush_bottom)
        painter.drawEllipse((self.width() - self.circle_radius) // 2, self.height() - self.padding - self.circle_radius, self.circle_radius, self.circle_radius)

class JoystickWidget(QWidget):
    def __init__(self, padding, radius):
        super().__init__()
        self.padding = padding
        self.circle_radius = radius

        self.setGeometry(0, 0, 700, 700)
        self.rx=0
        self.ry=0
        
        self.lx = 0
        self.ly = 0


    def paintEvent(self,event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

         # Set brush color for the outline (black)

        # Set pen color and width for the outline
        pen = QPen(QColor(Qt.white))
        pen.setWidth(2)  # Adjust the width as needed

        # Set brush style to NoBrush for no fill
        painter.setBrush(Qt.NoBrush)

        # Set the pen for drawing the outline
        painter.setPen(pen)

        painter.drawEllipse((self.width() - self.circle_radius) // 2, self.height()//2-self.circle_radius, self.circle_radius, self.circle_radius)
        painter.drawEllipse((self.width() - self.circle_radius) // 2, self.height()//2+self.padding + self.padding, self.circle_radius, self.circle_radius)

        painter.setBrush(Qt.white)
        painter.drawEllipse((self.circle_radius //2 + (self.width() - self.circle_radius) // 2) + self.rx, (self.circle_radius //2 + self.height()//2+self.padding + self.padding) + self.ry, 10, 10)
        painter.drawEllipse((self.circle_radius //2 + (self.width() - self.circle_radius) // 2) + self.lx, (self.height()//2 - self.circle_radius//2) + self.ly, 10, 10)

    def updateSticks(self, stick, x, y):
        if stick == "right_joystick":
            self.rx = x
            self.ry = y
        else:
            self.lx = x
            self.ly = y

        

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overlay")
        
    
        screen_geometry = QCoreApplication.instance().desktop().availableGeometry()
        x_position = screen_geometry.width() - self.width() +40
        y_position = 0

        self.setGeometry(x_position, y_position, 600, 1000)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.7)

        self.main_widget = QWidget()

        layout = QHBoxLayout()
        vertical_layout = QVBoxLayout()
        
        self.gamepad_buttons = CircleWidget(50, 70)
        self.joystick = JoystickWidget(10, 200)

    
        layout.addWidget(self.gamepad_buttons)
        layout.addWidget(self.joystick)

        vertical_layout.addLayout(layout)

        hide_button = QPushButton("Hide Window")
        hide_button.clicked.connect(self.close) 
        vertical_layout.addWidget(hide_button)


        # Center the button beneath the widgets
        vertical_layout.addWidget(hide_button, alignment=Qt.AlignCenter)

        self.main_widget.setLayout(vertical_layout)
        
        self.setCentralWidget(self.main_widget)

        


        

        

class Overlay():
    def __init__(self, *args):
        # Gamepad wrapper init
        self.wrapper = GamepadWrapper(self.handle_gamepad)
        self.wrapper.start()
        self.app = QApplication(list(args))
        self.window = MainWindow()
        self.app.aboutToQuit.connect(self.onclose)
        
        self.timer = QTimer()

       
        
        self.timer.timeout.connect(self.run_functions)
        # Set interval to 500 milliseconds (0.5 seconds)
        self.timer_interval = 1000

        self.timer.start(self.timer_interval)


    def start(self):
        self.window.show()
        sys.exit(self.app.exec())
        

    def onclose(self):
        self.wrapper.stop() # MUST STOP THE WRAPPER THREAD OR IT WILL RUN FOREVER


    def handle_gamepad(self, event):
        # print(event.event.__name__)
        event.run()
        if 'joystick' in event.event.__name__:
            self.window.joystick.updateSticks(event.event.__name__, event.info[event.event.__name__][1]["x_value"],event.info[event.event.__name__][1]["y_value"])
        elif 'button' in event.event.__name__:

            button_value = event.info[event.event.__name__][0][0]
            if 'press' in event.event.__name__:
                
                if button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_A:
                    self.window.gamepad_buttons.brush_bottom.setColor(Qt.green)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_B:
                    self.window.gamepad_buttons.brush_right.setColor(Qt.red)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_X:
                    self.window.gamepad_buttons.brush_left.setColor(Qt.blue)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_Y:
                    self.window.gamepad_buttons.brush_top.setColor(Qt.yellow)

            elif 'release' in event.event.__name__:
                if button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_A:
                    self.window.gamepad_buttons.brush_bottom.setColor(Qt.gray)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_B:
                    self.window.gamepad_buttons.brush_right.setColor(Qt.gray)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_X:
                    self.window.gamepad_buttons.brush_left.etColor(Qt.gray)
                elif button_value == vg.XUSB_BUTTON.XUSB_GAMEPAD_Y:
                    self.window.gamepad_buttons.brush_top.setColor(Qt.yellow)
            self.window.joystick.update()
            self.window.gamepad_buttons.update()
        self.wrapper.gamepad.update()

    def run_functions(self):
        self.test_on()
        # Start a new timer for 0.5 seconds to execute test_off function after test_on
        QTimer.singleShot(self.timer_interval//2, self.test_off)


    def test_on(self):
        ev = Event(self.wrapper.gamepad.right_joystick, x_value=50, y_value=0)
        ev2 = Event(self.wrapper.gamepad.left_joystick, x_value=0, y_value=50)
        ev3 = Event(self.wrapper.gamepad.press_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        self.wrapper.addEvent(ev)
        self.wrapper.addEvent(ev2)
        self.wrapper.addEvent(ev3)


    def test_off(self):
        ev = Event(self.wrapper.gamepad.right_joystick,  x_value=0, y_value=0)
        ev2 = Event(self.wrapper.gamepad.left_joystick, x_value=0, y_value=0)
        ev3 = Event(self.wrapper.gamepad.release_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        self.wrapper.addEvent(ev)
        self.wrapper.addEvent(ev2)
        self.wrapper.addEvent(ev3)
    
if __name__ == "__main__":
    app = Overlay(sys.argv)
    app.start()