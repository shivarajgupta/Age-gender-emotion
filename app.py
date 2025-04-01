import os
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

# Load Haar cascade for face detection
haarcascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

data_dir = "face_dataset"
os.makedirs(data_dir, exist_ok=True)

# Streamlit App
st.title("Face Recognition System")

# Step 1: Capture and Store Faces
def create_face_dataset(name, image):
    person_dir = os.path.join(data_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    img_path = os.path.join(person_dir, f"{name}_{len(os.listdir(person_dir)) + 1}.jpg")
    image.save(img_path)
    st.success(f"Image saved: {img_path}")

# Step 2: Train Face Dataset
def train_face_dataset():
    embeddings = {}
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            embeddings[person] = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embeddings[person].append(embedding)
                except Exception as e:
                    st.warning(f"Failed to process {img_path}: {e}")
    np.save("embeddings.npy", embeddings)
    st.success("Training completed and embeddings saved.")

# Step 3: Recognize Faces
def recognize_faces(image, embeddings):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = img_array[y:y + h, x:x + w]
        
        try:
            analysis = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
            analysis = analysis[0] if isinstance(analysis, list) else analysis
            age, gender, emotion = analysis["age"], analysis["gender"], max(analysis["emotion"], key=analysis["emotion"].get)
            
            face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            
            match, max_similarity = "Unknown", -1
            for person, person_embeddings in embeddings.items():
                for embed in person_embeddings:
                    similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
                    if similarity > max_similarity:
                        max_similarity, match = similarity, person
            
            label = f"{match} ({max_similarity:.2f})" if max_similarity > 0.7 else "Unknown"
            st.write(f"### {label}\nAge: {int(age)}, Gender: {gender}, Emotion: {emotion}")
        
        except Exception as e:
            st.warning(f"Error recognizing face: {e}")

# Streamlit UI Sections
option = st.sidebar.radio("Select an option", ["Capture Faces", "Train Dataset", "Recognize Faces"])

if option == "Capture Faces":
    name = st.text_input("Enter Name:")
    image = st.camera_input("Capture an Image")
    if image and name:
        img = Image.open(image)
        create_face_dataset(name, img)

elif option == "Train Dataset":
    if st.button("Train Now"):
        train_face_dataset()

elif option == "Recognize Faces":
    image = st.camera_input("Capture an Image for Recognition")
    if image and os.path.exists("embeddings.npy"):
        embeddings = np.load("embeddings.npy", allow_pickle=True).item()
        img = Image.open(image)
        recognize_faces(img, embeddings)
    else:
        st.warning("No embeddings found. Train the dataset first.")
