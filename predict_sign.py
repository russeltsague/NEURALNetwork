import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")

CATEGORIES = {  0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "End speed limit 80",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End all speed/passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing for vehicles over 3.5t"


}

def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    resized = cv2.resize(image, (30, 30))
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)
    
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return (predicted_class, CATEGORIES.get(predicted_class, "Unknown"), confidence)

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        
        self.label = tk.Label(root, text="Upload a traffic sign image")
        self.label.pack(pady=10)
        
        self.upload_btn = tk.Button(root, text="Browse Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        self.result_label = tk.Label(root, text="", font=('Helvetica', 14))
        self.result_label.pack(pady=10)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            
            prediction = predict_image(file_path)
            if prediction:
                class_id, class_name, confidence = prediction
                self.result_label.config(
                    text=f"Prediction: {class_name} (ID: {class_id})\nConfidence: {confidence:.2%}"
                )
            else:
                self.result_label.config(text="Error: Could not process image")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()