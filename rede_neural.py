import math
import numpy as np
from random import random, seed
from typing import List

from funcao_transferencia import FuncaoTransferencia

class RedeNeural:
    def __init__(
        self, 
        grid: List[List[float]], 
        classes: List[list[float]], 
        cabecalho: List[str], 
        quantidade_camada_oculta: int=0,
        quantidade_iteracoes: int = 2000,
        taxa_aprendizagem: int = 1,
        limiar_erro: float = 0.00001,
        funcao_saida = FuncaoTransferencia.linear,
        funcao_saida_derivada = FuncaoTransferencia.linear_derivada,
        ) -> None:
        # todos os dados, menos as classes
        self.grid = grid 

        # cabecalho ja com os novos nomes de acordo com o hot encode
        self.cabecalho: List[str] = cabecalho

        # coluna classes inteira, sem hot encode
        self.classes = classes 
        
        # quantidades entrada, saida, oculta (calculado na configuracao)
        self.quantidade_camada_entrada = 0
        self.quantidade_camada_saida = 0
        self.quantidade_camada_oculta = quantidade_camada_oculta

        # funções de calculo apos a somatoria nos neurónios
        self.funcao_saida = funcao_saida    
        self.funcao_saida_derivada = funcao_saida_derivada

        # quantidade e iteracoes que o algoritmo fará na aprendizagem
        self.quantidade_iteracoes = quantidade_iteracoes

        # limiar de parada de aprendizagem
        self.limiar_erro = limiar_erro
        
        # usado para calcular os novos pesos
        self.taxa_aprendizagem = taxa_aprendizagem

        # pesos da rede
        self.pesos_camada_oculta= []
        self.pesos_camada_saida= []

        self.__configurar()

    def __configurar(self):
        self.__quantidades_neuronios_camadas()
        self.__iniciar_rede()


    def __iniciar_rede(self):
        seed(1)

        # montar pesos camada oculta
        for _ in range(self.quantidade_camada_entrada):
            entrada = []
            for _ in range(self.quantidade_camada_oculta):
                valor_random = random()
                entrada.append(valor_random)
            self.pesos_camada_oculta.append(entrada)

        # montar pesos camada saida
        for _ in range(self.quantidade_camada_oculta):
            entrada = []
            for _ in range(self.quantidade_camada_saida):
                valor_random = random()
                entrada.append(valor_random)
            self.pesos_camada_saida.append(entrada)

    def treinar(self):

        # transform to numpy array instances
        self.pesos_camada_oculta = np.array(self.pesos_camada_oculta)
        self.pesos_camada_saida = np.array(self.pesos_camada_saida)

        saidas_camada_oculta = []
        saidas_camada_saida = []

        for linha in zip(self.grid):
            somas = np.dot(np.array(linha), self.pesos_camada_oculta) 
            saidas_camada_oculta = self.funcao_saida(somas)

            somas = np.dot(saidas_camada_oculta, self.pesos_camada_saida) 
            saidas_camada_saida = self.funcao_saida(somas)
            
            # calcular erro de saida
            # calcular erro camada oculta
            # atualizar pesos da camada de saida
            # atualizar pesos da camada de oculta


    def __quantidades_neuronios_camadas(self):
        self.quantidade_camada_entrada = len(self.grid[0])
        self.quantidade_camada_saida = len(self.cabecalho)
        
        calcular_quantidade_camada_oculta = lambda quantidade_camada_entrada, quantidade_camada_oculta: math.ceil((quantidade_camada_entrada + quantidade_camada_oculta) / 2)
        
        if self.quantidade_camada_oculta == 0:
            self.quantidade_camada_oculta = calcular_quantidade_camada_oculta(
                self.quantidade_camada_entrada, 
                self.quantidade_camada_oculta
            )
