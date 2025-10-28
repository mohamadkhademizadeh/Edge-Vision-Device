from pymodbus.client.tcp import ModbusTcpClient

class ModbusClient:
    def __init__(self, host='127.0.0.1', port=502, coil_address=1):
        self.client = ModbusTcpClient(host=host, port=port)
        self.coil_address = coil_address

    def write_pass_fail(self, ok: bool):
        self.client.connect()
        self.client.write_coil(self.coil_address, ok)
        self.client.close()
