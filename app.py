import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import cv2
import tempfile
from transformers import pipeline
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
from fer import FER
import speech_recognition as sr
from pydub import AudioSegment

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['OUTPUT_FOLDER'] = 'static/outputs/'

# Set up the emotion detector and models
emotion_detector = FER()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
caption_generator = pipeline("text2text-generation", model="facebook/bart-base")

# Ensure output folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Function to extract frames
def extract_frames(video_path, interval=1):
    frames = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    success, frame = video.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frames.append(frame)
        success, frame = video.read()
        count += 1
    video.release()
    return frames

# Function to detect emotions
def detect_emotions_in_frames(frames):
    emotion_data = []
    for frame in frames:
        emotion, score = emotion_detector.top_emotion(frame)
        if score is not None:
            emotion_data.append((frame, emotion, score))
    print(f"Detected {len(emotion_data)} emotions.")  # Debugging line
    return sorted(emotion_data, key=lambda x: x[2], reverse=True)[:5]

# Function to add title to images
def add_title_to_frame(image, title):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("arial.ttf", 40)
    text_position = (10, 10)  # Position at the top
    draw.text(text_position, title, fill="red", font=font)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Function to process video and generate title
def process_video(video_path):
    frames = extract_frames(video_path, interval=1)
    print(f"Extracted {len(frames)} frames.")  # Debugging line

    top_emotion_frames = detect_emotions_in_frames(frames)

    # Extract audio and transcribe
    audio_path = tempfile.mktemp(suffix=".wav")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    transcribed_text = recognizer.recognize_google(audio_data, language="en-US")

    # Summarize and generate title
    summary = summarizer(transcribed_text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
    title = caption_generator(summary, max_length=13)[0]['generated_text'].strip()

    # Save title and overlay on frames
    output_frames = []
    for idx, (frame, _, _) in enumerate(top_emotion_frames):
        titled_frame = add_title_to_frame(frame, title)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'top_emotion_frame_{idx}.jpg')
        print(f"Saving frame to {output_path}")  # Debugging line
        success = cv2.imwrite(output_path, titled_frame)
        if success:
            output_frames.append(output_path)
        else:
            print(f"Failed to save frame {idx}.")  # Debugging line
    return output_frames, title

# Route to display the main page and handle uploads
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the uploaded video
            output_frames, title = process_video(file_path)
            image_urls = [os.path.join('outputs', os.path.basename(image)) for image in output_frames]
            return render_template('index.html', image_urls=image_urls, title=title)
    return render_template('index.html', image_urls=[], title="")

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to serve processed output images
@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
