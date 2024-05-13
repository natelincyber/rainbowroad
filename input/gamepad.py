from threading import Thread
import vgamepad as vg
from collections import deque
import time

# queue based input scheduler

class Event():
    def __init__(self, event, *args, **kwargs):
        self.event = event
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.event(*self.args, **self.kwargs)



class GamepadWrapper(Thread):

    def __init__(self, callback):
        super().__init__()
        self.gamepad = vg.VX360Gamepad()
        self.callback = callback
        self.running = True
        self.buttonActions = deque()

    def run(self):
        while self.running:
            if not len(self.buttonActions) == 0:
                action = self.buttonActions.popleft()
                if action == "stop":
                    self.stop()
                else:
                    self.callback(action)
            
    def stop(self):
        self.buttonActions.clear()
        self.running = False
    
    # events are runnables
    def addEvent(self, event):
        self.buttonActions.append(event)



if __name__ == "__main__":
    def test_callback(action):
        action.run()
        wrapper.gamepad.update()


    wrapper = GamepadWrapper(test_callback)
    wrapper.start()
       
    ev = Event(wrapper.gamepad.left_joystick, x_value=-10000, y_value=0)
    ev2 = Event(wrapper.gamepad.left_joystick, x_value=0, y_value=0)

    for i in range(30):
        wrapper.addEvent(ev)
        time.sleep(0.1)
        wrapper.addEvent(ev2)
        time.sleep(0.1)

    wrapper.addEvent("stop")
     