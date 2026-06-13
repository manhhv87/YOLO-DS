import cv2
import os
from datetime import datetime
from pypylon import pylon


class BaslerCamera:
    def __init__(self, save_folder="raw_images"):
        self.cam = None
        self.converter = None
        self.save_folder = save_folder

        os.makedirs(self.save_folder, exist_ok=True)

    # -----------------------------
    # CONNECT CAMERA
    # -----------------------------
    def connect(self):
        try:
            self.cam = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.cam.Open()

            # Image converter
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            print("[Basler] Connected")
            return True
        except Exception as e:
            print("[Basler] Connect Error:", e)
            self.cam = None
            return False

    # -----------------------------
    # DISCONNECT CAMERA
    # -----------------------------
    def disconnect(self):
        try:
            if self.cam:
                self.cam.Close()
                print("[Basler] Disconnected")
            self.cam = None
        except Exception as e:
            print("[Basler] Disconnect Error:", e)

    # -----------------------------
    # CAPTURE ONE FRAME (RETURN IMAGE)
    # -----------------------------
    def grab(self, timeout=3000):
        if self.cam is None:
            print("[Basler] Error: Camera not connected.")
            return None

        grab = None
        try:
            self.cam.StartGrabbingMax(1)
            grab = self.cam.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                return self.converter.Convert(grab).GetArray()
            return None

        except Exception as e:
            print("[Basler] Grab Error:", e)
            return None

        finally:
            # Always release the grab result and stop grabbing so a single
            # timeout/exception cannot leave the camera wedged in a grabbing
            # state and break every subsequent capture.
            if grab is not None:
                grab.Release()
            try:
                if self.cam is not None and self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
            except Exception:
                pass

    # -----------------------------
    # CAPTURE + SAVE TO FOLDER
    # -----------------------------
    def capture_and_save(self, ext=".png"):
        img = self.grab()
        if img is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_folder, f"{timestamp}{ext}")

        try:
            cv2.imwrite(filename, img)
            return filename
        except Exception as e:
            print("[Basler] Save Error:", e)
            return None
