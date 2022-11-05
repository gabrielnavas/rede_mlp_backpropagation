from functools import reduce
import math
import numpy as np
from random import random, seed
from typing import List

from funcao_transferencia import FuncaoTransferencia

class RedeNeuralTreinamento:
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

        # erro camada_saida
        self.erros_camada_saida = []
        self.erros_camada_oculta = []

        # numero de epocas
        self.epocas = 0

        # erros de todas as linhas, por epocas 
        self.erros_rede = []

        self.__configurar()

    def __configurar(self):
        self.__configurar_quantidades_neuronios_camadas()
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

        # montar errors camada oculta
        for _ in range(self.quantidade_camada_oculta):
            self.erros_camada_oculta.append(0)

        # montar errors camada saida
        for _ in range(self.quantidade_camada_saida):
            self.erros_camada_saida.append(0)


    def treinar(self):
        # iniciar numero de epocas
        self.epocas = 0

        # transformar pesos em array numpy
        self.pesos_camada_oculta = np.array(self.pesos_camada_oculta)
        self.pesos_camada_saida = np.array(self.pesos_camada_saida)

        saidas_camada_oculta = []
        saidas_camada_saida = []
        media_erro_rede = self.limiar_erro + 1

        while media_erro_rede > self.limiar_erro and self.epocas < self.quantidade_iteracoes:

            for index_linha, linha in enumerate(self.grid):

                # somar e calcular saida CAMADA OCULTA
                somas_camada_oculta = np.dot(np.array(linha), self.pesos_camada_oculta) 
                saidas_camada_oculta = [self.funcao_saida(soma) for soma in somas_camada_oculta]

                # somar e calcular saida CAMADA SAIDAq
                somas_camada_saida = np.dot(np.array(saidas_camada_oculta), self.pesos_camada_saida) 
                saidas_camada_saida = [self.funcao_saida(soma) for soma in somas_camada_saida]
                
                # calcular erros CAMADA SAIDA
                saidas_esperada = self.classes[index_linha]
                for saida_esperada, (index_camada_saida, saida_camada_saida), soma_camada_saida in zip(saidas_esperada, enumerate(saidas_camada_saida), somas_camada_saida):
                    erro = (saida_esperada - saida_camada_saida) * self.funcao_saida_derivada(soma_camada_saida)
                    self.erros_camada_saida[index_camada_saida] = erro

                # calcular erro da rede, somatória erro quadratica * meio
                erro_rede_linha = self.__calcular_erro_rede_linha(self.erros_camada_saida)
                
                # adicionar na lista esse erro, linha por linha
                self.erros_rede.append(erro_rede_linha) 

                # calcular erros CAMADA OCULTA
                for index_camada_oculta, neuronio_camada_oculta in enumerate(self.pesos_camada_saida):
                    soma_erro_camada_oculta_n = 0.00
                    for index_camada_saida, peso_camada_oculta in enumerate(neuronio_camada_oculta):
                        erro_camada_saida = self.erros_camada_saida[index_camada_saida]
                        soma_camada_saida = somas_camada_saida[index_camada_saida]
                        erro_camada_oculta = (peso_camada_oculta * erro_camada_saida) * self.funcao_saida_derivada(soma_camada_saida)
                        soma_erro_camada_oculta_n += erro_camada_oculta

                    self.erros_camada_oculta[index_camada_oculta] = soma_erro_camada_oculta_n

                # calcular pesos CAMADA SAIDA
                for index_camada_oculta, neuronio_camada_oculta in enumerate(self.pesos_camada_saida):
                    for index_camada_saida, peso_camada_oculta in enumerate(neuronio_camada_oculta):
                        erro = self.erros_camada_saida[index_camada_saida]
                        entrada = saidas_camada_oculta[index_camada_oculta]
                        novo_peso = peso_camada_oculta + (self.taxa_aprendizagem * erro * entrada)
                        self.pesos_camada_saida[index_camada_oculta, index_camada_saida] = novo_peso

                # calcular pesos CAMADA OCULTA
                for index_camada_entrada, camada_entrada in enumerate(self.pesos_camada_oculta):
                    for index_camada_oculta, peso_camada_entrada in enumerate(camada_entrada):
                        erro = self.erros_camada_oculta[index_camada_oculta]
                        entrada = linha[index_camada_entrada]
                        novo_peso = peso_camada_entrada + (self.taxa_aprendizagem * erro * entrada)
                        self.pesos_camada_oculta[index_camada_entrada, index_camada_oculta] = novo_peso


            media_erro_rede = self.__calcular_media_erro_rede()
            self.epocas += 1

            print(self.epocas, media_erro_rede)

    def __configurar_quantidades_neuronios_camadas(self):
        self.quantidade_camada_entrada = len(self.grid[0])
        self.quantidade_camada_saida = len(self.classes[0])
        
        # calcular_quantidade_camada_oculta = lambda quantidade_camada_entrada, quantidade_camada_saida: (quantidade_camada_entrada + quantidade_camada_saida) / 2
        
        if self.quantidade_camada_oculta == 0:
            self.quantidade_camada_oculta = math.ceil(self.quantidade_camada_entrada + self.quantidade_camada_saida / 2)


    def __somatoria_erro_quadratica(self, total, erro):
        return total + (erro ** 2)

    def __calcular_erro_rede_linha(self, erros_camada_saida):
        somatoria_erro_quadratica = reduce(self.__somatoria_erro_quadratica, erros_camada_saida, 0.00)
        erro_rede = 0.5 * somatoria_erro_quadratica
        return erro_rede
        

    def __calcular_media_erro_rede(self):
        soma_erro = reduce(lambda total, erro: total+erro, self.erros_rede, 0.00)
        numero_erros = len(self.erros_rede)
        media_erro_rede = soma_erro / numero_erros
        return media_erro_rede
