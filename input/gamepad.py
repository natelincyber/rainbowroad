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
        self.info = {event.__name__: [self.args, self.kwargs]}
        

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
    def test_callback(action): # updates must be run in the callback until I figure smth out
        action.run()
        wrapper.gamepad.update()


    wrapper = GamepadWrapper(test_callback)
    wrapper.start()
       
    ev = Event(wrapper.gamepad.press_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    ev2 = Event(wrapper.gamepad.release_button, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    for i in range(10):
        wrapper.addEvent(ev)
        print(ev.info)
        time.sleep(0.1)
        wrapper.addEvent(ev2)
        print(ev2.info)
        time.sleep(0.1)

    wrapper.addEvent("stop")
     