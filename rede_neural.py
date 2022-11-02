import pprint
from typing import List

from hot_encode_coluna import HotEncodeColuna
from normalizacao import Normalizacao

from hot_encode_grid import HotEncodeGrid

class RedeNeural:
    def __init__(self, grid: List[List[float]], classes: List[str], cabecalho: List[str]) -> None:
        # todos os dados, menos as classes
        self.grid = grid 
        self.grid_normalizada_hot_encoded: List[List[float]] = []

        # nomes todos os cabecalhos da grid, sem hot encode
        self.cabecalho_grid = cabecalho 
        # cabecalho ja com os novos nomes de acordo com o hot encode
        self.cabecalho_grid_hot_encoded: List[str] = []

        # coluna classes inteira, sem hot encode
        self.classes = classes 
        # classes já transformado em binario
        self.classes_hot_encoded: List[list[float]] = []

    def configurar_classes(self):
        # TODO: colocar mais colunas classes no test.csv
        # TODO: colocar hot encode quando tiver mais de uma classe vai vir matrix NxM igual na grid
        # TODO: arrumar colunas de lista de lista em colunas normal
        hot_encode = HotEncodeColuna(coluna=self.classes)
        hot_encode.encode()
        self.classes_hot_encoded = hot_encode.coluna_encoded

    def configurar_atributos(self):
        # normalizar as colunas numero entre 0 e 1, que não for texto
        normalizacao = Normalizacao(grid=self.grid)
        normalizacao.normalizar()
        grid_normalizada = normalizacao.grid
        
        # aplicar hot encoded na grid ja normalizada 
        # verificando qual coluna é texto
        # nas colunas texto, aumentando a quantidade de colunas
        # TODO: arrumar colunas normalizadas de array em itens de lista
        hot_encode_grid = HotEncodeGrid(grid=grid_normalizada, cabecalho_grid=self.cabecalho_grid)
        hot_encode_grid.encode()
        self.grid_normalizada_hot_encoded = hot_encode_grid.grid
        self.cabecalho_grid_hot_encoded = hot_encode_grid.cabecalho_grid
        pprint.pprint(self.grid_normalizada_hot_encoded)