import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime

class FaceAndObjectDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Take Snapshot", width=15, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.btn_record = tk.Button(window, text="Start Recording", width=15, command=self.start_recording)
        self.btn_record.pack(anchor=tk.CENTER, expand=True)

        self.btn_stop = tk.Button(window, text="Stop Recording", width=15, command=self.stop_recording)
        self.btn_stop.pack(anchor=tk.CENTER, expand=True)
        self.btn_stop.config(state=tk.DISABLED)

        self.delay = 10
        self.faces_output_folder = "detected_faces"
        os.makedirs(self.faces_output_folder, exist_ok=True)
        self.objects_output_folder = "detected_objects"
        os.makedirs(self.objects_output_folder, exist_ok=True)

        self.is_recording = False
        self.output_video = None

        self.update()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            faces = self.detect_faces(frame)
            if faces.any():
                self.save_face_image(frame, faces)

            objects = self.detect_objects(frame)
            if objects:
                self.save_object_image(frame, objects)

    def start_recording(self):
        self.is_recording = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"recorded_video_{timestamp}.avi"
        self.output_video = cv2.VideoWriter(output_filename, fourcc, 20.0, (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        self.btn_record.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

    def stop_recording(self):
        self.is_recording = False
        self.output_video.release()
        self.btn_record.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            faces = self.detect_faces(frame)
            self.draw_faces(frame, faces)

            objects = self.detect_objects(frame)
            self.draw_objects(frame, objects)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            if self.is_recording:
                self.output_video.write(frame)

        self.window.after(self.delay, self.update)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def draw_faces(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    def save_face_image(self, frame, faces):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y+h, x:x+w]
            filename = os.path.join(self.faces_output_folder, f"face_{timestamp}_{i}.png")
            cv2.imwrite(filename, face_img)

    def detect_objects(self, frame):
        # Placeholder logic for object detection
        objects = [(50, 50, 100, 100), (200, 200, 50, 50)]
        return objects

    def draw_objects(self, frame, objects):
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def save_object_image(self, frame, objects):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (x, y, w, h) in enumerate(objects):
            object_img = frame[y:y+h, x:x+w]
            filename = os.path.join(self.objects_output_folder, f"object_{timestamp}_{i}.png")
            cv2.imwrite(filename, object_img)

def main():
    root = tk.Tk()
    app = FaceAndObjectDetectionApp(root, "Face and Object Detection App")
    root.mainloop()

if __name__ == "__main__":
    main()