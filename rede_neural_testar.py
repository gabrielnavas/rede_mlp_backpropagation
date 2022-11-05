import pprint
import numpy as np
from typing import List

class RedeNeuralTestar:
    def __init__(
        self, 
        grid: List[List[float]], 
        classes: List[list[float]], 
        nomes_classes: List[str], 
        pesos_camada_oculta: List[List[float]],
        pesos_camada_saida: List[List[float]],
        funcao_saida,
        funcao_saida_derivada,
        ) -> None:
        # todos os dados, menos as classes
        self.grid = grid 

        # nomes das classes ordenada por reconhecimento no arquivo de treinamento
        self.nomes_classes = nomes_classes

        # coluna classes inteira, com hot encode
        self.classes = classes 
        
        # quantidades entrada, saida, oculta (calculado na configuracao)
        self.quantidade_camada_entrada = 0
        self.quantidade_camada_saida = 0

        # funções de calculo apos a somatoria nos neurónios
        self.funcao_saida = funcao_saida    
        self.funcao_saida_derivada = funcao_saida_derivada
 
        # pesos da rede
        self.pesos_camada_oculta = pesos_camada_oculta
        self.pesos_camada_saida = pesos_camada_saida

        # matriz confusao NxN
        self.matriz_confusao = []


    def testar(self):
        self.__configurar_matrix_confusao()

        saidas_camada_oculta = []
        saidas_camada_saida = []

        for index_linha, linha in enumerate(self.grid):

            # somar e calcular saida CAMADA OCULTA
            somas_camada_oculta = np.dot(np.array(linha), self.pesos_camada_oculta) 
            saidas_camada_oculta = [self.funcao_saida(soma) for soma in somas_camada_oculta]

            # somar e calcular saida CAMADA SAIDAq
            somas_camada_saida = np.dot(np.array(saidas_camada_oculta), self.pesos_camada_saida) 
            saidas_camada_saida = [self.funcao_saida(soma) for soma in somas_camada_saida]
            

            saida_maior = max(saidas_camada_saida)
            index_saida_rede = saidas_camada_saida.index(saida_maior)
            
            saida_esperada = self.classes[index_linha]
            index_saida_esperada = saida_esperada.index(1)

            self.matriz_confusao[index_saida_esperada][index_saida_rede] += 1

            
    
    def obter_matriz_confusao(self):
        acuracia_por_classe = []
        somatoria_linhas = 0.00
        somatoria_diagonal = 0.00
        acuracia_total = 0.00
        erro_total = 0.00
        
        # calcular acuraria por classe
        for index in range(len(self.matriz_confusao)):
            valor_diagonal = self.matriz_confusao[index][index]
            total_linha = sum(self.matriz_confusao[index])
            somatoria_diagonal += valor_diagonal
            somatoria_linhas += total_linha
            acuracia_linha = valor_diagonal / total_linha
            acuracia_por_classe.append(acuracia_linha)

        acuracia_total = somatoria_diagonal / somatoria_linhas
        erro_total = 1 - acuracia_total

        return {
            "acuraria_por_classe": acuracia_por_classe,
            "acuraria_total": acuracia_total,
            "erro_total": erro_total,
            "matriz_confusao": self.matriz_confusao
        }

    def __configurar_matrix_confusao(self):
        self.matriz_confusao = []
        for index_linha in range(len(self.nomes_classes)):
            self.matriz_confusao.append([])
            for _ in range(len(self.nomes_classes)):  
                self.matriz_confusao[index_linha].append(0)