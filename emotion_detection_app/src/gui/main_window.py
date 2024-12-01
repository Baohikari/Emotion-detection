import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tf_keras.models import load_model
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
            self.update_frame()

    def stop_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.cap:
                self.cap.release()
            self.camera_label.config(image='')

    def update_frame(self):
        if self.is_camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
            self.camera_label.after(10, self.update_frame)


if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()