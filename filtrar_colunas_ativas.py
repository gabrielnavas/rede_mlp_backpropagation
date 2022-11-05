import pprint
from typing import List

import numpy as np

class FiltrarColunasAtivas:
    def __init__(
        self, 
        grid: List[List[float]], 
        colunas_grid_ativa: List[bool]):

        # todos os dados, menos as classes
        self.grid = grid 
        self.colunas_grid_ativa = colunas_grid_ativa

    def filtrar(self):
        grid = np.array(self.grid)
        numero_colunas_retiradas = 0
        for index_coluna_ativa, coluna_ativa in enumerate(self.colunas_grid_ativa):
            if not coluna_ativa:
                grid = np.delete(grid, obj=index_coluna_ativa-numero_colunas_retiradas, axis=1)
                numero_colunas_retiradas += 1

        self.grid = []
        for linha in grid:
            nova_linha = np.ndarray.tolist(linha)
            self.grid.append(nova_linha)
