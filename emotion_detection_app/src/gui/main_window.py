import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
from tf_keras.models import load_model
import pyaudio
import threading
import librosa
import pickle
import wave

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Nhận diện cảm xúc khuôn mặt')
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f'{screen_width}x{screen_height}')

        #Tạo tiêu đề cho ứng dụng
        self.title_label = tk.Label(self,
                                    text='Ứng dụng nhận diện cảm xúc',
                                    font=('calibri', 20, 'bold'),
                                    )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky='nsew')

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        #Tải mô hình cảm xúc
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_path = os.path.join(project_root, 'models', 'best_emotion_model.h5')
        print("Đường dẫn là: ", model_path)
        self.emotion_model = load_model(model_path)
        #Các cảm xúc
        self.emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprised', 'Fearful', 'Disgusted', 'Neutral']
        #Tạo các text widget cho row1, row2, row3
        self.text_widget_intro = tk.Text(self,
                                          bg='mistyrose',
                                          fg='black',
                                          height=10,
                                          width=10)
        self.text_widget_intro.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.text_widget_intro.insert('end', 'Ứng dụng nhận diện cảm xúc\n', 'bold')
        self.text_widget_intro.tag_config('bold', font=('calibri', 16, 'bold'))
        self.text_widget_intro.config(state='disabled')

        self.text_widget_members = tk.Text(self,
                                            bg='mistyrose',
                                            fg='black',
                                            height=10,
                                            width=10)
        self.text_widget_members.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        self.text_widget_members.insert('end', 'Nhóm 2\n', 'bold')
        self.text_widget_members.insert('end', 'Lớp học phần: Đồ án chuyên ngành\n', 'normal')
        self.text_widget_members.insert('end', 'Mã lớp học phần: 010100086404 \n', 'normal')
        self.text_widget_members.tag_config('bold', font=('calibri', 16, 'bold'))
        self.text_widget_members.config(state='disabled')

        self.text_widget_use = tk.Text(self,
                                        bg='mistyrose',
                                        fg='black',
                                        height=10,
                                        width=10)
        self.text_widget_use.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')
        self.text_widget_use.insert('end', 'Cách sử dụng', 'bold')
        self.text_widget_use.tag_config('bold', font=('calibri', 16, 'bold'))
        self.text_widget_use.config(state='disabled')

        #Gộp cột 2 từ row1 đến row3 để chứa camera
        self.camera_frame = tk.Frame(self,
                                     bd=2,
                                     bg='pink',
                                     relief='groove')
        self.camera_frame.grid(row=1, column=1, rowspan=3, padx=10, pady=10, sticky='nsew')
        
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(expand=True, fill='both')

        #Row 4, cột 1: Chia thành 3 cột nhỏ hơn cho các button, tất cả nằm trong cột 1
        self.button_frame = tk.Frame(self)
        
        self.button_frame.grid(row=4, column=1, padx=10, pady=10, sticky='n')

        self.button1 = tk.Button(self.button_frame,
                                 text='Dừng',
                                 font=('calibri', 12),
                                 bd=2,
                                 relief='groove',
                                 height=2,
                                 command=self.stop_camera)
        
        self.button2 = tk.Button(self.button_frame,
                                 text='Bắt đầu',
                                 font=('calibri', 12),
                                 bd=2,
                                 relief='groove', 
                                 height=2,
                                 command=self.start_camera)
        
        self.button3 = tk.Button(self.button_frame,
                                 text='Lưu kết quả',
                                 font=('calibri', 12),
                                 bd=2,
                                 relief='groove',
                                 height=2)
        
        #Đặt các nút vào trong khung với layout 3 cột nhỏ
        self.button1.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.button2.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        self.button3.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')

        #Cấu hình để 3 cột này có kích thước bằng nhau
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)

        #Cấu hình các hàng và cột để mở rộng đều nhau
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)

        self.cap = None
        self.is_camera_running = False

    def start_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            self.is_camera_running = True
            self.audio_thread = threading.Thread(target=self.start_audio_recording)
            self.audio_thread.start()
            self.update_frame()

    def stop_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.cap:
                self.cap.release()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.camera_label.config(image='')
    def start_audio_recording(self):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=44100,
                             input=True,
                             frames_per_buffer=1024)
        self.audio_frames = []
        while self.is_camera_running:
            data = self.stream.read(1024)
            self.audio_frames.append(data)
    
    def process_audio_and_predict_emotion(self):
        if self.audio_frames:
            audio_data = b''.join(self.audio_frames)
            temp_wav_path = "temp_audio.wav"
            with wave.open(temp_wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(audio_data)
            y, sr = librosa.load(temp_wav_path, sr=44100)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            audio_model_path = os.path.join(project_root, 'models', 'audio_emotion_model.pkl')
            with open(audio_model_path, 'rb') as f:
                audio_model = pickle.load(f)
            
            predicted_emotion = audio_model.predict([mfccs_mean])[0]

            # Hiển thị kết quả
            print(f"Cảm xúc giọng nói: {predicted_emotion}")
            self.text_widget_use.config(state='normal')
            self.text_widget_use.insert('end', f"\nCảm xúc giọng nói: {predicted_emotion}", 'normal')
            self.text_widget_use.config(state='disabled')
            
        os.remove(temp_wav_path)

    def update_frame(self):
        if self.is_camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Phát hiện khuôn mặt
                faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Cắt khuôn mặt ra để phân tích cảm xúc
                    face_roi = frame[y:y + h, x:x + w]
                    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    face_roi_resized = cv2.resize(face_roi_rgb, (48, 48))  # Resize về kích thước phù hợp với mô hình

                    # Tiền xử lý ảnh khuôn mặt
                    face_roi_normalized = face_roi_resized / 255.0
                    face_roi_gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_RGB2GRAY)  # Chuyển ảnh về grayscale
                    face_roi_reshaped = np.expand_dims(face_roi_gray, axis=0)  # Thêm batch dimension
                    face_roi_reshaped = np.expand_dims(face_roi_reshaped, axis=-1)  # Thêm channel dimension (1)

                    # Dự đoán cảm xúc từ khuôn mặt
                    emotion_prediction = self.emotion_model.predict(face_roi_reshaped)
                    max_index = np.argmax(emotion_prediction[0])
                    predicted_emotion = self.emotion_labels[max_index]

                    # Hiển thị cảm xúc dự đoán lên khung hình
                    cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)

            self.camera_label.after(10, self.update_frame)
