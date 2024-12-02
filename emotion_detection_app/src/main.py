import sys
import os

# Lấy đường dẫn thư mục gốc của dự án (src là thư mục gốc của mã nguồn)
project_root = os.path.abspath(os.path.dirname(__file__))

# Thêm thư mục gốc vào sys.path để Python có thể tìm thấy thư mục src
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import MainWindow từ gui.main_window
from gui.main_window import MainWindow

def main():
    app = MainWindow()
    app.mainloop()

if __name__ == '__main__':
    main()
