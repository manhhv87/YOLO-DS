import time
from opcua import ua
from core.opc_client import OPCUAClient
from core.pipeline import main_pipeline, run_without_camera


# ============= CONFIG =============
OPC_URL = "opc.tcp://127.0.0.1:49320"   # sửa lại
NODE_REQ    = "ns=2;s=Channel1.Device1.bit_REQ"
NODE_BUSY   = "ns=2;s=Channel1.Device1.bit_BUSY"
NODE_RESULT = "ns=2;s=Channel1.Device1.word_RESULT"
# ==================================


def vision_loop():
    opc = OPCUAClient(OPC_URL)

    print("Connecting OPC UA...")
    if not opc.connect():
        print("Cannot connect OPC")
        return

    print("OPC connected")

    # Ensure BUSY = 0, RESULT = 0 on start
    opc.write_bit(NODE_BUSY, 0)
    opc.write_word(NODE_RESULT, 0)

    print("Vision server ready.")

    while True:
        time.sleep(0.05)

        # Read REQ from PLC
        req = opc.read(NODE_REQ)

        # If PLC requests inspection
        if req == 1:
            print("PLC → Request received")

            # Set BUSY = 1
            opc.write_bit(NODE_BUSY, 1)

            # Chạy pipeline thật
            result_img, result_info = main_pipeline()

            # Xác định OK/NG
            if result_info["status"] == "OK":
                overall = 1  # OK
            else:
                overall = 2  # NG

            # Gửi result về PLC
            opc.write_word(NODE_RESULT, overall)
            print("Vision → RESULT =", overall)

            # Clear BUSY
            opc.write_bit(NODE_BUSY, 0)

            print("Vision done. Waiting PLC reset REQ → 0")

            # Wait PLC reset REQ
            while opc.read(NODE_REQ) == 1:
                time.sleep(0.05)

            print("PLC reset detected, ready for next cycle.")


if __name__ == "__main__":
    vision_loop()
