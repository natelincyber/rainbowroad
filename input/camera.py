import cv2
from threading import Thread
from collections import deque

# camera wrapper for multithreaded image grabbing and caching
class CameraWrapper(Thread):
    def __init__(self, **kwargs):
        super().__init__()
        self.running = True

        self.camera_source = 0
        for item in kwargs:
            if item == "camera_source":
                self.camera_source = kwargs[item]
        self.camera = None
        self.frames = deque()

        self.camera_init()


    # creates camera oject
    def camera_init(self):
        self.camera = cv2.VideoCapture(self.camera_source)
        if not self.camera.isOpened():
            raise ValueError(f"Unable to open camera source {self.camera_source}")
    
        
    # THIS IS THE ONLY FUNCTION THAT THE USER SHOULD CALL!!!!
    # returs the most recent frame
    def get_frame(self):
        if len(self.frames) != 0:
            return self.frames.popleft()
        else:
            return []

    def run(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frames.append(frame)

    def stop(self):
        self.running = False




if __name__ == "__main__":
    wrapper = CameraWrapper(camera_source=0)
    wrapper.start()
    while True:
        frame = wrapper.get_frame()
        if len(frame)!=0:
            cv2.imshow("asd", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            wrapper.stop()
            break
















