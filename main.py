import tkinter as tk
from tkinter import messagebox as mess
import cv2, os, csv, numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from tkinter import simpledialog as tsd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure opencv-contrib-python is installed for face module
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("Face module is not available. Please install opencv-contrib-python.")

# Helper functions
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def contact():
    mess.showinfo(title='Contact Us', message="Contact us at: 'your-email@example.com'")

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror(
            title='Missing File',
            message="Please ensure 'haarcascade_frontalface_default.xml' is in the same directory."
        )
        window.destroy()

def save_pass():
    new_pass = tsd.askstring('Password', 'Enter new password:', show='*')
    if new_pass is None:
        return
    confirm_pass = tsd.askstring('Password', 'Confirm password:', show='*')
    if new_pass == confirm_pass:
        with open("password.txt", "w") as f:
            f.write(new_pass)
        mess.showinfo(title='Success', message="Password changed successfully!")
    else:
        mess.showerror(title='Error', message="Passwords do not match!")

def load_name_mapping():
    name_mapping = {}
    if os.path.exists("StudentDetails/StudentDetails.csv"):
        with open("StudentDetails/StudentDetails.csv", "r") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                name_mapping[int(row[0])] = row[1]
    return name_mapping

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(image_np)
        ids.append(id)
    return faces, ids

def save_performance_graphics(true_ids, predicted_ids):
    cm = confusion_matrix(true_ids, predicted_ids)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    assure_path_exists("Performance")
    plt.savefig("Performance/confusion_matrix.png")
    plt.close()

def save_metrics_to_csv(metrics, save_path="Performance/metrics.csv"):
    headers = ["Accuracy", "Precision", "Recall", "F1-Score"]
    assure_path_exists("Performance")
    with open(save_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(metrics)

def save_qualitative_results(frame, name, conf, save_path="QualitativeResults"):
    assure_path_exists(save_path)
    file_name = f"{save_path}/{name}_conf_{int(conf)}.png"
    cv2.imwrite(file_name, frame)

def evaluate_model(test_faces, test_ids, recognizer):
    predictions = []
    for face in test_faces:
        predicted_id, _ = recognizer.predict(face)
        predictions.append(predicted_id)

    acc = accuracy_score(test_ids, predictions)
    prec = precision_score(test_ids, predictions, average='weighted', zero_division=0)
    rec = recall_score(test_ids, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_ids, predictions, average='weighted', zero_division=0)

    save_performance_graphics(test_ids, predictions)
    save_metrics_to_csv([acc, prec, rec, f1])

    return acc, prec, rec, f1

# Functions for GUI buttons
def take_images():
    id = txt.get()
    name = txt2.get()
    if not id or not name:
        mess.showerror("Error", "Please enter both ID and Name!")
        return

    assure_path_exists("TrainingImage")
    assure_path_exists("StudentDetails")
    cam = cv2.VideoCapture(0)
    harcascade_path = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascade_path)

    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"TrainingImage/{name}.{id}.{count}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Collecting Images: {count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Taking Images", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    with open("StudentDetails/StudentDetails.csv", "a+", newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([id, name])
    mess.showinfo("Success", f"Images collected for {name}!")

def train_images():
    assure_path_exists("TrainingImageLabel")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels("TrainingImage")
    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")

    # Evaluate the model
    acc, prec, rec, f1 = evaluate_model(faces, ids, recognizer)
    mess.showinfo("Model Trained", f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}")

def track_images():
    assure_path_exists("Attendance")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("TrainingImageLabel/Trainner.yml")
    except cv2.error:
        mess.showerror("Error", "Model not trained yet!")
        return

    harcascade_path = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascade_path)

    cam = cv2.VideoCapture(0)
    name_mapping = load_name_mapping()
    col_names = ['ID', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                name = name_mapping.get(id, "Unknown")
                attendance.loc[len(attendance)] = [id, name, datetime.date.today(), time.strftime('%H:%M:%S')]
                save_qualitative_results(img, name, conf)

        cv2.imshow("Face", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    attendance.to_csv(f"Attendance/Attendance_{datetime.date.today()}.csv", index=False)
    mess.showinfo("Attendance", "Attendance saved successfully!")

# GUI Setup
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("900x700")
window.configure(bg="#3A3B3C")

# Fonts
title_font = ('Segoe UI', 18, 'bold')
label_font = ('Segoe UI', 14, 'bold')
button_font = ('Segoe UI', 12)

# Header
header_frame = tk.Frame(window, bg="#00A9A6", bd=0)
header_frame.pack(fill=tk.X)
header_label = tk.Label(header_frame, text="Face Recognition Attendance System", bg="#00A9A6", fg="white", font=title_font)
header_label.pack(pady=20)

# Widgets
lbl = tk.Label(window, text="Enter ID", bg="#3A3B3C", fg="white", font=label_font)
lbl.place(x=100, y=180)
txt = tk.Entry(window, font=label_font, bd=2)
txt.place(x=250, y=180)

lbl2 = tk.Label(window, text="Enter Name", bg="#3A3B3C", fg="white", font=label_font)
lbl2.place(x=100, y=250)
txt2 = tk.Entry(window, font=label_font, bd=2)
txt2.place(x=250, y=250)

# Buttons
take_img = tk.Button(window, text="Take Images", bg="#00A9A6", fg="white", font=button_font, command=take_images)
take_img.place(x=100, y=350)

train_img = tk.Button(window, text="Train Images", bg="#00A9A6", fg="white", font=button_font, command=train_images)
train_img.place(x=250, y=350)

track_img = tk.Button(window, text="Track Images", bg="#00A9A6", fg="white", font=button_font, command=track_images)
track_img.place(x=400, y=350)

# Run
check_haarcascadefile()
window.mainloop()
mess.showinfo("Model Trained", f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}")
