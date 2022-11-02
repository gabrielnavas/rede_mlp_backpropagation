from typing import List


class HotEncodeColuna:
   def __init__(self, coluna: List[str | int | float]) -> None:
      # coluna nao encodada
      self.coluna = coluna

      # coluna encodada
      self.coluna_encoded: List[List[int]] = []

      # nome não repitidos
      self.nomes = []

      # flag para marcar se é texto a coluna
      try:
         float(self.coluna[0])
         self.is_text_column = False
      except:
         self.is_text_column = True
   
   def __pegar_nomes_classes(self):
      # pega todos os nomes das classes, sem repitir
      for nome in self.coluna:
         if not nome in self.nomes:
            self.nomes.append(nome)

   def __hot_encode_classes(self):
      # somente faz o hot encode se for maior que 2
      if len(self.nomes) <= 2:
         return 
      
      # adicionar nomes das classes nos cabecalhos
      for nome in self.coluna:
         #pegar o index na nome
         index_nome = self.nomes.index(nome)

         nomes_hot_encoded = [
            1 if index == index_nome else 0 
            for index, _ in enumerate(range(len(self.nomes)))]
         self.coluna_encoded.append(nomes_hot_encoded)
   
   def encode(self):
      self.__pegar_nomes_classes()
      self.__hot_encode_classes()