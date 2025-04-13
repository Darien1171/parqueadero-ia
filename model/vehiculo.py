class vehiculo():
    def __init__(self, placa, marca, modelo, soat_numero, soat_vencimiento, tecno_numero, tecno_vencimiento, id_usuario):
        self.placa = placa
        self.marca = marca
        self.modelo = int(modelo)
        self.soat_numero = soat_numero
        self.soat_vencimiento = soat_vencimiento
        self.tecno_numero = tecno_numero
        self.tecno_vencimiento = tecno_vencimiento
        self.idUsuario = int(id_usuario)