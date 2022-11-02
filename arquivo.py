class Arquivo:
    def __init__(self, path: str) -> None:
        self.path = path
        self.cabecalho = []
        self.grid = []
        self.classes = []

    def tornar_itens_grid_numericos(self):
        for index_x, atributos in enumerate(self.grid):
            # obtem range de atributos
            for index_y, valor in enumerate(atributos):
                # verifica se é numérico
                try:
                    self.grid[index_x][index_y] = float(valor)
                except Exception as ex:
                    self.grid[index_x][index_y] = valor

    def ler_arquivo(self):
        with open(self.path, 'r') as file:
            linhas=file.readlines()
            # obtem a cabecalho
            self.cabecalho = linhas[0].replace('\n', '').split(',')

            # obtem a grid
            self.grid = [linha.replace('\n', '').split(',')[0:-1] for linha in linhas[1:]]
            
            # pega as classe
            self.classes = [linha.replace('\n', '').split(',')[-1] for linha in linhas[1:]]
