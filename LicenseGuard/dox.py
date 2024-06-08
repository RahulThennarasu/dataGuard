import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import numpy as np
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Create an OCR reader
reader = easyocr.Reader(['en'])

# Function to blur personal information in the image
def blur_personal_info(img, text_to_blur):
    # Use EasyOCR to perform OCR and get bounding boxes
    results = reader.readtext(img)

    # Process each detected text region
    for result in results:
        bbox, text = result[0], result[1]
        bbox = [(int(point[0]), int(point[1])) for point in bbox]

        # Check if the text matches the text to blur
        if text_to_blur.lower() in text.lower():
            ymin, ymax, xmin, xmax = (
                min(point[1] for point in bbox),
                max(point[1] for point in bbox),
                min(point[0] for point in bbox),
                max(point[0] for point in bbox),
            )
            roi = img[ymin:ymax, xmin:xmax]
            roi = cv2.GaussianBlur(roi, (31, 31), 0)  # Adjust kernel size for desired blur
            img[ymin:ymax, xmin:xmax] = roi

    return img

# Function to check if the text contains personal information
def is_personal_info(text):
    # Tokenize the text using NLTK
    words = word_tokenize(text)
    
    # Part-of-speech tagging
    tagged_words = pos_tag(words)
    
    # Named Entity Recognition
    named_entities = ne_chunk(tagged_words)
    
    # Check for named entities such as PERSON, DATE, LOCATION, etc.
    for entity in named_entities:
        if hasattr(entity, 'label') and entity.label() in ['PERSON', 'DATE', 'GPE']:
            return True
    
    return False

# Function to open file dialog and load image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and display the original image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(img)
        tk_image_original = ImageTk.PhotoImage(original_image)
        label_original.config(image=tk_image_original)
        label_original.image = tk_image_original
        label_original_width = label_original.winfo_reqwidth()
        label_original.place(x=screen_width/2-label_original_width/2, y=200)

        # Prompt for the text to blur
        blur_text_window = tk.Toplevel(root)
        blur_text_window.title("Enter Text to Blur")

        blur_text_label = tk.Label(blur_text_window, text="Enter text to blur:")
        blur_text_label.pack(pady=10)

        blur_text_entry = tk.Entry(blur_text_window)
        blur_text_entry.pack(pady=5)

        blur_button = tk.Button(blur_text_window, text="Blur", command=lambda: process_and_display_image(file_path, blur_text_entry.get()))
        blur_button.pack(pady=10)

# Global variable to store the image with ongoing blur effect
blurred_img = None

# Function to process and display the blurred image
def process_and_display_image(file_path, text_to_blur):
    global blurred_img
    
    img = cv2.imread(file_path)
    if blurred_img is None:
        blurred_img = img.copy()
    
    # Apply previous blur effect to the new image
    img = blurred_img.copy()
    
    # Apply blur effect to the new text to blur
    img = blur_personal_info(img, text_to_blur)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Blur faces
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        face_roi = cv2.GaussianBlur(face_roi, (31, 31), 0)
        img[y:y+h, x:x+w] = face_roi
    
    blurred_img = img.copy()  # Update the global blurred image
    
    # Display the blurred image
    blurred_image = Image.fromarray(img)
    tk_image_blurred = ImageTk.PhotoImage(blurred_image)
    label_original.config(image=tk_image_blurred)
    label_original.image = tk_image_blurred

# Create the main window
root = tk.Tk()
root.title("License Guard")

# Create and configure widgets
screen_width = root.winfo_screenwidth()

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)
button_width = upload_button.winfo_reqwidth()
upload_button.place(x=screen_width/2-button_width/2, y=100)

label = tk.Label(
    root,
    text="LicenseGuard",
    font=("Arial", 30),  # Set the font
    fg="black",          # Set the text color
)
label.pack()
label_width = label.winfo_reqwidth()
label.place(x=screen_width/2-label_width/2, y=30)

label_original = tk.Label(root)
label_original.pack(pady=10)

# Center the window on the screen
root.eval('tk::PlaceWindow . center')

# Start Tkinter main loop
root.mainloop()
