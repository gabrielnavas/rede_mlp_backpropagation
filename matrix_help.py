from typing import List

class MatrixHelp:
   
   @staticmethod
   def obter_colunas(matrix: List[List[any]]):
      coluna_index = 0
      for coluna_y, _ in enumerate(matrix[0]):
         coluna = []
         for linha_x in range(len(matrix)):
            coluna.append(matrix[linha_x][coluna_y])
         # retorna uma coluna de cada vez
         yield coluna, coluna_index
         coluna_index += 1
