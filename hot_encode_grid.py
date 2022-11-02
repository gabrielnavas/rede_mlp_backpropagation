from typing import List

from matrix_help import MatrixHelp
from hot_encode_coluna import HotEncodeColuna


class HotEncodeGrid:
   def __init__(self, grid: List[List[str | int | float]], cabecalho_grid: List[List[str]]) -> None:
      self.grid = grid
      self.cabecalho_grid = cabecalho_grid

   def __transformar_grid_lista_linear(self):
      for index_x, linha in enumerate(self.grid):
         nova_linha = []
         for valor_ou_lista in linha:
            if isinstance(valor_ou_lista, list):
               for valores in valor_ou_lista:
                  nova_linha.append(valores)
            else:
               nova_linha.append(valor_ou_lista)
         self.grid[index_x] = nova_linha
            
   def __transformar_cabecalho_lista_linear(self):
      nova_linha = []
      for valor_ou_lista in self.cabecalho_grid:
         if isinstance(valor_ou_lista, list):
            for valores in valor_ou_lista:
               nova_linha.append(valores)
         else:
            nova_linha.append(valor_ou_lista)
      self.cabecalho_grid = nova_linha

   def encode(self):
      for coluna, coluna_y in MatrixHelp.obter_colunas(self.grid):
         hot_encode_coluna = HotEncodeColuna(coluna=coluna)
         if hot_encode_coluna.is_text_column:
            hot_encode_coluna.encode()
            
            # adiciona essa coluna no lugar da coluna de texto
            for linha_x, _ in enumerate(range(len(self.grid))):
               self.grid[linha_x][coluna_y] = hot_encode_coluna.coluna_encoded[linha_x]
               self.cabecalho_grid[coluna_y] = hot_encode_coluna.nomes

      self.__transformar_grid_lista_linear()
      self.__transformar_cabecalho_lista_linear()
