import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from tqdm import tqdm
import threading
import queue
from detect_objects import SuspiciousObjectDetector  # ✅ Ensure this is correct

class SuspiciousObjectDetectorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Suspicious Object Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f3f4f6')

        # ✅ Queue before starting the thread
        self.queue = queue.Queue()

        # Initialize detector in a separate thread
        self.detector = None
        self.init_thread = threading.Thread(target=self.initialize_detector)
        self.init_thread.start()

        # Video variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.is_playing_raw = False  # ✅ Added for raw play
        self.current_frame = None

        self.create_ui()
        self.update_loading_status()

        # Clean exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def initialize_detector(self):
        try:
            self.detector = SuspiciousObjectDetector()  # ✅ Corrected reference
            self.queue.put(("success", "AI Model loaded successfully!"))
        except Exception as e:
            self.queue.put(("error", f"Failed to load AI Model: {str(e)}"))

    def create_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="AI-Powered Suspicious Object Detection",
            font=("Helvetica", 24, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Loading indicator
        self.loading_frame = ttk.Frame(self.main_frame)
        self.loading_frame.pack(fill=tk.X, pady=20)
        self.loading_label = ttk.Label(
            self.loading_frame,
            text="Loading AI Model...",
            font=("Helvetica", 12)
        )
        self.loading_label.pack()
        self.progress = ttk.Progressbar(
            self.loading_frame,
            mode='indeterminate'
        )
        self.progress.pack(fill=tk.X, pady=10)
        self.progress.start()

        # Video frame
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        self.canvas = tk.Canvas(
            self.video_frame,
            bg='black',
            width=800,
            height=450
        )
        self.canvas.pack(pady=20)

        # Controls frame
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.pack(fill=tk.X, pady=20)

        self.upload_btn = ttk.Button(
            controls_frame,
            text="Upload Video",
            command=self.upload_video,
            state=tk.DISABLED
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(
            controls_frame,
            text="Analyze Video",
            command=self.start_analysis,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(
            controls_frame,
            text="Play",
            command=self.toggle_play,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.play_raw_btn = ttk.Button(  # ✅ New Button
            controls_frame,
            text="Play Raw Video",
            command=self.toggle_play_raw,
            state=tk.DISABLED
        )
        self.play_raw_btn.pack(side=tk.LEFT, padx=5)

        # Results frame
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        self.results_tree = ttk.Treeview(
            self.results_frame,
            columns=("Time", "Object", "Status", "Confidence"),
            show="headings"
        )

        self.results_tree.heading("Time", text="Timestamp")
        self.results_tree.heading("Object", text="Object")
        self.results_tree.heading("Status", text="Status")
        self.results_tree.heading("Confidence", text="Confidence")

        self.results_tree.pack(fill=tk.BOTH, expand=True)

    def update_loading_status(self):
        try:
            msg_type, message = self.queue.get_nowait()
            self.progress.stop()
            self.loading_frame.pack_forget()

            if msg_type == "success":
                self.upload_btn.configure(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", message)
        except queue.Empty:
            self.root.after(100, self.update_loading_status)

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.analyze_btn.configure(state=tk.NORMAL)
            self.play_btn.configure(state=tk.NORMAL)
            self.play_raw_btn.configure(state=tk.NORMAL)  # ✅ Enabled Raw Button
            self.show_frame()

    def show_frame(self):
        if self.cap and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame

                # 🚀 Detect objects in the current frame
                detections = self.detector.detect_objects(frame, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

                # 🚀 Draw detections
                frame = self.detector.draw_detections(frame, detections)

                # Convert frame to RGB for Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame.thumbnail((800, 450))
                photo = ImageTk.PhotoImage(frame)
                self.canvas.create_image(
                    400, 225,
                    image=photo,
                    anchor=tk.CENTER
                )
                self.canvas.photo = photo
                self.root.after(30, self.show_frame)
            else:
                self.is_playing = False
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.play_btn.configure(text="Play")

    def show_raw_frame(self):  # ✅ New Raw Frame Function
        if self.cap and self.is_playing_raw:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame

                # 🚀 Just show the raw frame (no detection)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame.thumbnail((800, 450))
                photo = ImageTk.PhotoImage(frame)
                self.canvas.create_image(
                    400, 225,
                    image=photo,
                    anchor=tk.CENTER
                )
                self.canvas.photo = photo
                self.root.after(30, self.show_raw_frame)
            else:
                self.is_playing_raw = False
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.play_raw_btn.configure(text="Play Raw Video")

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.is_playing_raw = False  # ✅ Stop raw when normal play
        self.play_btn.configure(
            text="Pause" if self.is_playing else "Play"
        )
        self.play_raw_btn.configure(text="Play Raw Video")
        if self.is_playing:
            self.show_frame()

    def toggle_play_raw(self):  # ✅ New toggle play raw function
        self.is_playing_raw = not self.is_playing_raw
        self.is_playing = False  # ✅ Stop normal when raw play
        self.play_raw_btn.configure(
            text="Pause Raw" if self.is_playing_raw else "Play Raw Video"
        )
        self.play_btn.configure(text="Play")
        if self.is_playing_raw:
            self.show_raw_frame()

    def analyze_video(self):
        try:
            results = self.detector.process_video(self.video_path)
            self.root.after(0, self.update_results, results)
        except Exception as error:
            self.root.after(0, lambda e=error: messagebox.showerror(
                "Error",
                f"Analysis failed: {str(e)}"
            ))
        finally:
            self.root.after(0, lambda: self.analyze_btn.configure(state=tk.NORMAL))

    def start_analysis(self):
        if not self.video_path or not self.detector:
            return

        self.analyze_btn.configure(state=tk.DISABLED)
        self.results_tree.delete(*self.results_tree.get_children())

        thread = threading.Thread(target=self.analyze_video)
        thread.start()

    def update_results(self, results):
        for result in results:
            status = result['status'].capitalize()
            confidence = f"{result['confidence'] * 100:.1f}%"
            timestamp = f"{result['timestamp']:.1f}s"

            self.results_tree.insert(
                "",
                tk.END,
                values=(
                    timestamp,
                    result['object'],
                    status,
                    confidence
                ),
                tags=(status.lower(),)
            )

        self.results_tree.tag_configure(
            'suspicious',
            background='#fee2e2',
            foreground='#991b1b'
        )
        self.results_tree.tag_configure(
            'safe',
            background='#dcfce7',
            foreground='#166534'
        )

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SuspiciousObjectDetectorUI()
    app.run()
