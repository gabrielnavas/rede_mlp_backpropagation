from typing import List

from matrix_help import MatrixHelp


class Normalizacao:
   def __init__(self, grid: List[List[str | int | float]]):
      self.grid = grid
      self.valores_max_coluna: List[float] = []
      self.valores_min_coluna: List[float] = []

   def __tentar_normalizar(self, valor, min_valor, max_valor):
      try:
         return (valor-min_valor) / (max_valor - min_valor)
      except:
         return valor

   def normalizar(self):
      for coluna, coluna_y in MatrixHelp.obter_colunas(self.grid):
         valor_min = min(coluna)
         valor_max = max(coluna)

         for linha_x, valor in enumerate(coluna):
            novo_valor = self.__tentar_normalizar(valor,valor_min, valor_max)
            self.grid[linha_x][coluna_y] = novo_valor

