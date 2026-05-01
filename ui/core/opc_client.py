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
        if self.connected:
            self.client.disconnect()
        self.connected = False

    def read(self, node):
        if not self.connected: return None
        return self.client.get_node(node).get_value()

    def write_word(self, node, value):
        if not self.connected: return False
        dv = ua.DataValue(ua.Variant(value, ua.VariantType.UInt16))
        self.client.get_node(node).set_value(dv)
        return True

    def write_bit(self, node, value):
        if not self.connected: return False
        dv = ua.DataValue(ua.Variant(value, ua.VariantType.Boolean))
        self.client.get_node(node).set_value(dv)
        return True


