from opcua import Client, ua

class OPCUAClient:
    def __init__(self, url):
        self.url = url
        self.client = None
        self.connected = False

    def connect(self):
        try:
            self.client = Client(self.url)
            self.client.connect()
            self.connected = True
            return True
        except:
            self.connected = False
            return False

    def disconnect(self):
        try:
            if self.connected and self.client is not None:
                self.client.disconnect()
        except Exception as e:
            print("[OPC] disconnect error:", e)
        finally:
            self.connected = False

    # All I/O is wrapped so a transient server/network error returns a safe
    # default instead of propagating up and killing the VisionWorker thread
    # (which would hang the line).
    def read(self, node):
        if not self.connected: return None
        try:
            return self.client.get_node(node).get_value()
        except Exception as e:
            print("[OPC] read error:", e)
            return None

    def write_word(self, node, value):
        if not self.connected: return False
        try:
            dv = ua.DataValue(ua.Variant(value, ua.VariantType.UInt16))
            self.client.get_node(node).set_value(dv)
            return True
        except Exception as e:
            print("[OPC] write_word error:", e)
            return False

    def write_bit(self, node, value):
        if not self.connected: return False
        try:
            dv = ua.DataValue(ua.Variant(value, ua.VariantType.Boolean))
            self.client.get_node(node).set_value(dv)
            return True
        except Exception as e:
            print("[OPC] write_bit error:", e)
            return False


