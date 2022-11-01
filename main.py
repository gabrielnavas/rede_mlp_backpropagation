from random import seed
from random import random

import pprint
from typing import List

class Normalizacao:
   def __init__(self, grid: List[List[str | int | float]]):
      self.grid = grid
      self.valores_max_coluna: List[float] = []
      self.valores_min_coluna: List[float] = []

   def __obter_colunas_grid(self):
      coluna_index = 0
      for coluna_y, _ in enumerate(self.grid[0]):
         coluna = []
         for linha_x in range(len(self.grid)):
            coluna.append(self.grid[linha_x][coluna_y])
         # retorna uma coluna de cada vez
         yield coluna, coluna_index
         coluna_index += 1

   
   def __tentar_normalizar(self, valor, min_valor, max_valor):
      try:
         return (valor-min_valor) / (max_valor - min_valor)
      except:
         return valor

   def normalizar(self):
      for coluna, coluna_y in self.__obter_colunas_grid():
         valor_min = min(coluna)
         valor_max = max(coluna)

         for linha_x, valor in enumerate(coluna):
            novo_valor = self.__tentar_normalizar(valor,valor_min, valor_max)
            self.grid[linha_x][coluna_y] = novo_valor


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

class FuncaoTranferencia:
   def linear(self, net):
      return net/10

   def linear_derivada(self, net):
      return 1/10

   def logistica(self, net):
      raise 'falta implementar'

   def logistica_linear(self, net):
      raise 'falta implementar'

   def linear_logistica(self, net):
      raise 'falta implementar'

   def linear_logistica_derivada(self, net):
      raise 'falta implementar'

class RedeNeural:
   def __init__(self, grid: List[List[float]], classes: List[str], cabecalho: List[str]) -> None:
      # todos os dados, menos as classes
      self.grid = grid 
      self.grid_hot_encoded: List[str] = []

      # nomes todos os cabecalhos sem hot encode
      self.cabecalho = cabecalho 

      # coluna classes inteira
      self.classes = classes 

      # nomes das classes sem hot encode
      self.cabecalho_classes: List[str] = []

      # classes já transformado em binario
      self.classes_hot_encoded: List[str] = []

   def __pegar_nomes_classes(self):
      # pega todos os nomes das classes, sem repitir
      for classe in self.classes:
         if not classe in self.cabecalho_classes:
            self.cabecalho_classes.append(classe)

   def __hot_encode_classes(self):
      # somente faz o hot encode se for maior que 2
      if len(self.classes) <= 2:
         return 
      
      # adicionar nomes das classes nos cabecalhos
      for classe in self.classes:
         #pegar o index na classe
         index_classe = self.cabecalho_classes.index(classe)

         classe_hot_encoded = [
            1 if index == index_classe else 0 
            for index, _ in enumerate(range(len(self.cabecalho_classes)))]
         self.classes_hot_encoded.append(classe_hot_encoded)

   

   def configurar_classes(self):
      self.__pegar_nomes_classes()
      self.__hot_encode_classes()
   

   def configurar_atributos(self):
      # normalizar as colunas numero entre 0 e 1, que não for texto
      normalizacao = Normalizacao(grid=dados.grid)
      normalizacao.normalizar()
      self.grid_hot_encoded = normalizacao.grid
      
      # hot encoded nas colunas texto, aumentando a quantidade de colunas



def iniciar_rede(bias, n_inputs, n_hidden, n_outputs):
   seed(1)

   rede = list()
   bias = 1 if bias else 0
   
   camada_oculta = [{'peso':[random() for i in range(n_inputs + bias)]} for i in range(n_hidden)]
   rede.append(camada_oculta)
   
   camada_saida = [{'peso':[random() for i in range(n_hidden + bias)]} for i in range(n_outputs)]
   rede.append(camada_saida)
   
   return rede



def somar_net_n(rede, entradas, camada_n):
   camada = rede[camada_n]

   for no in camada:
      soma = 0.00
      for peso, entrada in zip(no['peso'], entradas):
         soma += float(entrada * peso)
      no['net'] = soma   
   
   return rede

def calcular_saida_camada_oculta_n(rede, func_saida, camada_n):
   camada_oculta = rede[camada_n]
   
   for no in camada_oculta:
      saida = func_saida(no['net'])
      no['saida'] = saida

   return rede



def calc_erro_saida(rede, saidas_esperada, calc_saida_derivada):
   camada_saida = rede[-1]

   for no, saida_esperada in zip(camada_saida, saidas_esperada):
      erro_n = (saida_esperada - no['saida']) * calc_saida_derivada(no['saida'])
      no['erro'] = erro_n

   return rede

def continuar_treinando(rede):
   return True

def calc_erro_saida_n(rede, camada_n, calc_saida_derivada):
   camada_atual = rede[camada_n]
   camada_proxima = rede[camada_n + 1]

   for index_camada_atual, no_atual in enumerate(camada_atual):
      erro_no_atual = 0.00
      for no_proximo in camada_proxima:
         erro_no_atual += float(no_proximo['peso'][index_camada_atual] * no_proximo['erro'] * calc_saida_derivada(no_atual['net']))
      no_atual['erro'] = erro_no_atual

   return rede

def calc_novos_pesos(rede, taxa_aprendizagem, entradas, camada_n):
   camada = rede[camada_n]
   
   for no in camada:
      for (index_peso, peso), entrada in zip(enumerate(no['peso']), entradas):
         novo_peso = peso + (taxa_aprendizagem * no['erro'] * entrada)
         no['peso'][index_peso] = novo_peso

   return rede

def treinar_rede(rede, dados, taxa_aprendizagem):
   taxa_aprendizagem = 1

   for linha in dados:

      saida_esperada = linha[-1]
      linha = linha[0:-1]

      # soma net e saida da camada de saida
      rede = somar_net_n(rede, linha, camada_n=0)
      rede = calcular_saida_camada_oculta_n(rede, FuncaoTranferencia().linear, camada_n=0)

      # pega a saida e que é a entrada do proximo
      saida_anterior = []
      for no in rede[0]:
         saida_anterior.append(no['saida'])

      # soma net e saida da camada de saida
      rede = somar_net_n(rede, saida_anterior, camada_n=1)
      rede = calcular_saida_camada_oculta_n(rede, FuncaoTranferencia().linear, camada_n=1)

      # calcula o erro de saida
      rede = calc_erro_saida(rede, saida_esperada, FuncaoTranferencia().linear_derivada)

      # verifica o erro geral da rede
      if not continuar_treinando(rede):
         break

      # calcular erro camada oculta
      rede = calc_erro_saida_n(rede, camada_n=0)

      # calcular os novos pesos camada saida
      rede = calc_novos_pesos(rede, taxa_aprendizagem=taxa_aprendizagem, entradas=saida_anterior, camada_n=1)
      
      # calcular os novos pesos camada oculta
      rede = calc_novos_pesos(rede, taxa_aprendizagem=taxa_aprendizagem, entradas=linha, camada_n=0)

      pprint.pprint(rede, indent=4)

# if __name__ == '__main__':    
#    n_inputs = 2
#    n_hidden = 4
#    n_outputs = 1

#    rede = iniciar_rede(bias=False, n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)
   
#    dados = [
#       [0, 1, [1, 1]]
#    ]

#    rede = treinar_rede(rede=rede, dados=dados, taxa_aprendizagem=1)

if __name__ == '__main__':

   # recebe arquivo de algum lugar
   dados = Arquivo(path='/home/neo/Faculdade/TERMO_7/IA/2_BIM/TRABALHO/exemplo_base_dados/test.csv')
   
   # le o arquivo
   dados.ler_arquivo()

   # torna a grid numerica, se der
   dados.tornar_itens_grid_numericos()


   # inicia a rede
   rede = RedeNeural(
      grid=dados.grid, 
      cabecalho=dados.cabecalho, 
      classes=dados.classes
   ) 

   # configura as classes (hot encoded etc)
   rede.configurar_classes()
   
   # configura os atributos
   rede.configurar_atributos()

   # pprint.pprint([', '.join([str(b) for b in a]) for a in rede_iniciada.classes_hot_encoded], indent=4)
   pprint.pprint(rede.classes_hot_encoded)
   