from PyQt5 import QtWidgets, QtCore, QtGui
import cv2, os
import time
from core.camera import BaslerCamera
from core.opc_client import OPCUAClient

# ---------------------------------------------------------------------------
# Detector backend selector. Set USE_DS_YOLO=1 in the environment to swap the
# legacy 8-class YOLOv8m pipeline for the new DS-YOLO (binary OK/NG)
# pipeline. Defaults to the legacy backend so existing deployments are not
# disturbed by an accidental import.
# ---------------------------------------------------------------------------
if os.environ.get("USE_DS_YOLO", "0") == "1":
    from core.pipeline_ds import main_pipeline_ds as main_pipeline
    main_pipeline_no_shift = main_pipeline   # DS-YOLO needs no separate variant
    _DETECTOR = "DS-YOLO (binary OK/NG)"
else:
    from core.pipeline import main_pipeline, main_pipeline_no_shift
    _DETECTOR = "YOLOv8m (legacy 8-class)"
print(f"[window_camera] detector backend: {_DETECTOR}")

OPC_URL = "opc.tcp://127.0.0.1:49320"

#OPC Nodes
NODE_REQ    = "ns=2;s=Channel1.Device1.bit_REQ"
NODE_BUSY   = "ns=2;s=Channel1.Device1.bit_BUSY"
NODE_RESULT = "ns=2;s=Channel1.Device1.word_RESULT"
NODE_SENSOR = "ns=2;s=Channel1.Device1.bit_SENSOR"   # cảm biến tại vị trí phân loại


#IMAGE VIEWER
class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._zoom = 0
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def set_image(self, pixmap):
        self._photo.setPixmap(pixmap)
        self.fitInView()   # giữ fit lúc load ảnh

    def fitInView(self):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if rect.isNull():
            return
        self.setSceneRect(rect)
        self.resetTransform()
        viewrect = self.viewport().rect()
        scenerect = rect
        factor = min(viewrect.width() / scenerect.width(),
                     viewrect.height() / scenerect.height())
        self.scale(factor, factor)
        self._zoom = 0

    # ===============================
    #         FIX ZOOM HERE
    # ===============================
    def wheelEvent(self, event):
        if self._photo.pixmap().isNull():
            return

        zoom_in_factor = 1.25
        zoom_out_factor = 0.8

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        if self._zoom < -10:   # giới hạn zoom out
            self._zoom = -10
            return

        self.scale(zoom_factor, zoom_factor)


class VisionWorker(QtCore.QThread):
    update_image = QtCore.pyqtSignal(QtGui.QPixmap)
    update_result = QtCore.pyqtSignal(str)
    update_status = QtCore.pyqtSignal(str)

    update_cycle = QtCore.pyqtSignal(float)  # ms

    def __init__(self, opc, camera):
        super().__init__()
        self.opc = opc
        self.cam = camera          # camera được truyền từ AOIWindow
        self.running = True
        self.result_buffer = []
        self.processing = False    # pipeline đang chạy?

    # ======================================================
    # GỬI RESULT TỪ BUFFER → PLC
    # ======================================================
    def try_send_result(self):
        if not self.result_buffer:
            return

        if self.opc.read(NODE_SENSOR) != 1:
            return

        if self.opc.read(NODE_RESULT) != 0:
            return

        result = self.result_buffer.pop(0)
        self.opc.write_word(NODE_RESULT, result)
        self.update_status.emit(f"RESULT sent = {result}")

    # ======================================================
    # CHỤP ẢNH
    # ======================================================
    def capture_image(self):
        img_path = None

        for _ in range(3):
            img_path = self.cam.capture_and_save(".bmp")
            if img_path:
                break
            time.sleep(0.1)

        if img_path is None:
            raise Exception("Camera capture failed")

        return img_path

    # ======================================================
    # XỬ LÝ PIPELINE
    # ======================================================

    def run_pipeline(self, image_path):

        start = time.time()

        try:
            img, info = main_pipeline_no_shift(image_path)
        except Exception as e:
            self.result_buffer.append(2)
            self.update_status.emit(f"PIPELINE ERROR → NG | {str(e)}")
            self.processing = False
            return

        cv2.imwrite("_temp_result.png", img)
        self.update_image.emit(QtGui.QPixmap("_temp_result.png"))

        result = 1 if info["status"] == "OK" else 2
        self.result_buffer.append(result)
        self.update_result.emit("OK" if result == 1 else "NG")

        self.update_status.emit(f"Buffered RESULT = {result}")

        cycle_ms = (time.time() - start) * 1000
        self.update_cycle.emit(cycle_ms)

        self.processing = False

    # ======================================================
    # MAIN LOOP
    # ======================================================
    def run(self):
        self.opc.write_bit(NODE_BUSY, 0)
        self.opc.write_word(NODE_RESULT, 0)

        self.update_status.emit("Vision ready — waiting REQ...")

        pool = QtCore.QThreadPool.globalInstance()

        while self.running:
            QtCore.QThread.msleep(20)

            # gửi result nếu đủ điều kiện
            self.try_send_result()

            if self.processing:
                continue

            if self.opc.read(NODE_REQ) != 1:
                continue

            # ==========================
            # CHỤP ẢNH
            # ==========================
            self.update_status.emit("REQ=1 → capturing...")
            self.opc.write_bit(NODE_BUSY, 1)

            try:
                time.sleep(3)
                img_path = self.capture_image()
            except Exception as e:
                self.result_buffer.append(2)
                self.opc.write_bit(NODE_BUSY, 0)
                self.update_status.emit(f"CAPTURE ERROR → NG | {str(e)}")
                continue

            self.opc.write_bit(NODE_BUSY, 0)
            self.update_status.emit("Capture done → BUSY=0")

            # ==========================
            # CHẠY PIPELINE Ở THREADPOOL
            # ==========================
            self.processing = True

            ### Nếu dùng thật thì thay test_path = img_path
            #test_path = "C:/Users/vuong/PycharmProjects/smd-detect-python39/stored_images/raw_images/20251208_172312.bmp"
            self.run_pipeline(img_path)

            # ==========================
            # CHỜ PLC RESET REQ
            # ==========================
            while self.opc.read(NODE_REQ) == 1:
                QtCore.QThread.msleep(20)

            self.update_status.emit("REQ reset → ready next cycle")

#AOI MAIN WINDOW
class AOIWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI System (Auto Mode – PLC Controlled)")
        self.resize(1400, 800)

        self.opc = None
        self.worker = None

        self.count_ok = 0
        self.count_ng = 0
        self.count_total = 0

        self.start_time = None
        self.last_cycle = 0

        #UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        #LEFT PANEL
        left = QtWidgets.QFrame()
        left.setFixedWidth(220)
        left_layout = QtWidgets.QVBoxLayout(left)

        self.btn_connect = QtWidgets.QPushButton("Kết nối OPC UA")
        self.btn_disconnect = QtWidgets.QPushButton("Ngắt kết nối")
        self.btn_start = QtWidgets.QPushButton("Start AOI Server")

        for b in [self.btn_connect, self.btn_disconnect, self.btn_start]:
            b.setMinimumHeight(45)
            left_layout.addWidget(b)
        left_layout.addStretch()

        #MIDDLE IMAGE VIEW
        middle = QtWidgets.QFrame()
        mid_layout = QtWidgets.QVBoxLayout(middle)
        self.image_view = ImageViewer()
        mid_layout.addWidget(self.image_view)

        #RIGHT PANEL
        right = QtWidgets.QFrame()
        right.setFixedWidth(250)
        right_layout = QtWidgets.QVBoxLayout(right)

        self.lbl_plc_status = QtWidgets.QLabel("PLC: -")
        self.lbl_result = QtWidgets.QLabel("Result: -")

        # --- COUNTERS ---
        self.lbl_count_ok = QtWidgets.QLabel("OK: 0")
        self.lbl_count_ng = QtWidgets.QLabel("NG: 0")
        self.lbl_count_total = QtWidgets.QLabel("Total: 0")

        self.lbl_count_ok.setStyleSheet(
            "font-size:18px; font-weight:bold; color:red; background:#333; padding:8px;"
        )
        self.lbl_count_ng.setStyleSheet(
            "font-size:18px; font-weight:bold; color:lime; background:#333; padding:8px;"
        )
        self.lbl_count_total.setStyleSheet(
            "font-size:18px; font-weight:bold; color:#00aaff; background:#333; padding:8px;"
        )

        right_layout.addWidget(self.lbl_count_ok)
        right_layout.addWidget(self.lbl_count_ng)
        right_layout.addWidget(self.lbl_count_total)

        self.lbl_cycle = QtWidgets.QLabel("Cycle: - ms")
        self.lbl_cycle.setStyleSheet("font-size:18px;font-weight:bold;color:#ffaa00;background:#333;padding:8px;")
        right_layout.addWidget(self.lbl_cycle)

        self.lbl_uph = QtWidgets.QLabel("UPH: -")
        self.lbl_uph.setStyleSheet("font-size:18px;font-weight:bold;color:#ffaa00;background:#333;padding:8px;")
        right_layout.addWidget(self.lbl_uph)

        for w in [self.lbl_plc_status, self.lbl_result]:
            w.setStyleSheet("font-size:18px; font-weight:bold; color:white; background:#444; padding:10px;")
            right_layout.addWidget(w)
        right_layout.addStretch()

        #STATUS BAR
        self.status_bar = QtWidgets.QLabel("Hệ thống AOI: Đang chờ...")
        self.statusBar().addWidget(self.status_bar)

        #LAYOUT MERGE
        main_layout.addWidget(left)
        main_layout.addWidget(middle)
        main_layout.addWidget(right)

        #CONNECT SIGNALS
        self.btn_connect.clicked.connect(self.connect_opc)
        self.btn_disconnect.clicked.connect(self.disconnect_opc)
        self.btn_start.clicked.connect(self.start_server)

    #OPC FUNCTIONS
    def connect_opc(self):
        self.opc = OPCUAClient(OPC_URL)

        if self.opc.connect():
            self.lbl_plc_status.setText("PLC: Connected")
            self.lbl_plc_status.setStyleSheet("font-size:18px; font-weight:bold;color:green;background:#444;padding:10px;")
            self.status_bar.setText("Đã kết nối OPC UA.")
        else:
            self.status_bar.setText("Không kết nối được OPC UA!")

    def disconnect_opc(self):
        if self.worker:
            self.worker.running = False
            self.worker.quit()

        if self.opc:
            self.opc.disconnect()

        self.lbl_plc_status.setText("PLC: Disconnected")
        self.lbl_plc_status.setStyleSheet("font-size:18px; font-weight:bold;color:red;background:#444;padding:10px;")
        self.status_bar.setText("Đã ngắt kết nối OPC UA.")

    # =======================================================
    #                     START SERVER LOOP
    # =======================================================
    def start_server(self):
        if self.opc is None or not self.opc.connected:
            self.status_bar.setText("Chưa kết nối OPC UA!")
            return

        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        save_dir = os.path.join(BASE_DIR, "stored_images/raw_images")

        # camera khởi tạo ở đây (không phải trong VisionWorker)
        self.camera = BaslerCamera(save_folder=save_dir)
        self.camera.connect()

        self.worker = VisionWorker(self.opc, self.camera)

        self.worker.update_image.connect(self.image_view.set_image)
        self.worker.update_result.connect(self.update_result)
        self.worker.update_status.connect(self.status_bar.setText)

        self.worker.update_cycle.connect(self.update_cycle)

        self.worker.start()

        self.start_time = time.time()

        self.status_bar.setText("AOI server running (waiting PLC)...")

    def update_result(self, status):
        if status == "OK":
            self.lbl_result.setText("Result: OK")
            self.lbl_result.setStyleSheet("font-size:20px;font-weight:bold;color:green;background:#222;padding:10px;")

            self.count_ok += 1

        else:
            self.lbl_result.setText("Result: NG")
            self.lbl_result.setStyleSheet("font-size:20px;font-weight:bold;color:red;background:#222;padding:10px;")

            self.count_ng += 1

        self.count_total += 1

        # cập nhật UI
        self.lbl_count_ok.setText(f"OK: {self.count_ok}")
        self.lbl_count_ng.setText(f"NG: {self.count_ng}")
        self.lbl_count_total.setText(f"Total: {self.count_total}")

    def update_cycle(self, ms):
        self.last_cycle = ms
        self.lbl_cycle.setText(f"Cycle: {ms:.1f} ms")
        self.update_uph()

    def update_uph(self):
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return

        hours = elapsed / 3600
        uph = self.count_total / hours
        self.lbl_uph.setText(f"UPH: {uph:.1f}")


#MAIN APP
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    gui = AOIWindow()
    gui.show()
    app.exec_()
