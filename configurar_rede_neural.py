from typing import List

from filtrar_colunas_ativas import FiltrarColunasAtivas
from hot_encode_coluna import HotEncodeColuna
from normalizacao import Normalizacao
from hot_encode_grid import HotEncodeGrid


class ConfigurarDados:
    def __init__(
        self, 
        grid: List[List[float]], 
        colunas_grid_ativa: List[bool],
        classes: List[str], 
        cabecalho: List[str], 
        ) -> None:

        # todos os dados, menos as classes
        self.grid = grid 
        self.colunas_grid_ativa = colunas_grid_ativa
        self.grid_normalizada_hot_encoded: List[List[float]] = []

        # nomes todos os cabecalhos da grid, sem hot encode
        self.cabecalho_grid = cabecalho 
        # cabecalho ja com os novos nomes de acordo com o hot encode
        self.cabecalho_grid_hot_encoded: List[str] = []

        # coluna classes inteira, sem hot encode
        self.classes = classes 
        # classes já transformado em binario
        self.classes_hot_encoded: List[list[float]] = []
        
    def configurar(self):
        self.__configurar_classes()
        self.__filtrar_colunas_atributos_ativas()
        self.__configurar_atributos()

    def __filtrar_colunas_atributos_ativas(self):
        filtrar_colunas_ativas = FiltrarColunasAtivas(grid=self.grid, colunas_grid_ativa=self.colunas_grid_ativa)
        filtrar_colunas_ativas.filtrar()
        self.grid = filtrar_colunas_ativas.grid 

    def __configurar_classes(self):
        hot_encode = HotEncodeColuna(coluna=self.classes)
        hot_encode.encode()
        self.classes_hot_encoded = hot_encode.coluna_encoded

    def __configurar_atributos(self):
        normalizacao = Normalizacao(grid=self.grid)
        normalizacao.normalizar()
        grid_normalizada = normalizacao.grid
        
        hot_encode_grid = HotEncodeGrid(grid=grid_normalizada, cabecalho_grid=self.cabecalho_grid)
        hot_encode_grid.encode()
        self.grid_normalizada_hot_encoded = hot_encode_grid.grid
        self.cabecalho_grid_hot_encoded = hot_encode_grid.cabecalho_grid
