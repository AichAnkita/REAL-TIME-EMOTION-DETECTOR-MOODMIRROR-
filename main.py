import datetime
import threading
from collections import deque, Counter

import customtkinter as ctk
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from deepface import DeepFace
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Configs ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# --- App Setup ---
app = ctk.CTk()
app.geometry("1000x700")
app.title("Real-Time Mood Tracker")

video_running = True
current_mood = "Detecting..."

# Store recent moods for stability
emotion_history = deque(maxlen=15)
time_stamps = []
mood_log = []

# Mood to Emoji mapping
mood_emoji = {
    "Happy": "😄", "Sad": "😢", "Angry": "😠", "Surprise": "😲",
    "Fear": "😨", "Disgust": "🤢", "Neutral": "😐"
}

# --- UI Components ---
mood_label = ctk.CTkLabel(app, text="Mood: Detecting...", font=("Arial", 24))
mood_label.pack(pady=10)

video_label = ctk.CTkLabel(app, text="")
video_label.pack(pady=10)

graph_frame = ctk.CTkFrame(app)
graph_frame.pack(pady=10, fill="both", expand=True)


def update_graph():
    """Update mood over time graph"""
    if len(time_stamps) != len(mood_log) or len(mood_log) < 2:
        return

    # Map moods to integers
    unique_moods = sorted(set(mood_log))
    mood_to_num = {mood: idx for idx, mood in enumerate(unique_moods)}
    mood_nums = [mood_to_num[m] for m in mood_log]

    # Create figure and clear previous widgets
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(time_stamps, mood_nums, marker='o', linestyle='-', color='lime')
    ax.set_yticks(list(mood_to_num.values()))
    ax.set_yticklabels(list(mood_to_num.keys()))
    ax.set_title("Mood Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mood")
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart = FigureCanvasTkAgg(fig, master=graph_frame)
    chart.draw()
    chart.get_tk_widget().pack()


def update_mood(frame):
    global current_mood

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        detected_mood = result[0]['dominant_emotion'].capitalize()
        emotion_history.append(detected_mood)

        most_common = Counter(emotion_history).most_common(1)[0][0]
        current_mood = most_common
        emoji = mood_emoji.get(current_mood, "")

        # Update GUI elements in main thread
        app.after(0, lambda: mood_label.configure(text=f"Mood: {current_mood} {emoji}"))

        # Append mood log and timestamp safely
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        mood_log.append(current_mood)
        time_stamps.append(current_time)

        # Trim to avoid mismatch
        if len(mood_log) > 30:
            mood_log.pop(0)
            time_stamps.pop(0)

        app.after(0, update_graph)

    except Exception as e:
        print("Detection Error:", e)


def video_loop():
    cap = cv2.VideoCapture(0)

    while video_running:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update video in main thread
        app.after(0, lambda img=imgtk: update_video(img))

        # Run mood detection in a thread
        threading.Thread(target=update_mood, args=(frame,)).start()
        cv2.waitKey(200)

    cap.release()


def update_video(imgtk):
    video_label.configure(image=imgtk)
    video_label.image = imgtk


def on_closing():
    global video_running
    video_running = False
    app.destroy()


# --- Start Thread and GUI ---
video_thread = threading.Thread(target=video_loop, daemon=True)
video_thread.start()

app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()
