import pprint
from typing import List

from hot_encode_coluna import HotEncodeColuna
from normalizacao import Normalizacao

from hot_encode_grid import HotEncodeGrid

from funcao_transferencia import FuncaoTransferencia

class ConfigurarRedeNeural:
    def __init__(
        self, 
        grid: List[List[float]], 
        classes: List[str], 
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
        self.grid_normalizada_hot_encoded: List[List[float]] = []

        # nomes todos os cabecalhos da grid, sem hot encode
        self.cabecalho_grid = cabecalho 
        # cabecalho ja com os novos nomes de acordo com o hot encode
        self.cabecalho_grid_hot_encoded: List[str] = []

        # coluna classes inteira, sem hot encode
        self.classes = classes 
        # classes já transformado em binario
        self.classes_hot_encoded: List[list[float]] = []
        
        # # quantidades entrada, saida, oculta (calculado na configuracao)
        # self.quantidade_camada_entrada = 0
        # self.quantidade_camada_saida = 0
        # self.quantidade_camada_oculta = quantidade_camada_oculta

        # # rede em sí,com a camada de entrada, oculta e saida sendo configurada
        # self.rede = []

        # # funções de calculo apos a somatoria nos neurónios
        # self.funcao_saida = funcao_saida    
        # self.funcao_saida_derivada = funcao_saida_derivada

        # # quantidade e iteracoes que o algoritmo fará na aprendizagem
        # self.quantidade_iteracoes = quantidade_iteracoes

        # # limiar de parada de aprendizagem
        # self.limiar_erro = limiar_erro
        
        # # usado para calcular os novos pesos
        # self.taxa_aprendizagem = taxa_aprendizagem

    def configurar(self):
        self.__configurar_classes()
        self.__configurar_atributos()
        # self.__quantidades_neuronios_camadas()


    # def iniciar_rede(self):
    #     seed(1)

    #     self.rede = list()
        
    #     camada_oculta = [{'peso':[random() for i in range(self.quantidade_camada_entrada)]} for i in range(self.quantidade_camada_oculta)]
    #     self.rede.append(camada_oculta)
        
    #     camada_saida = [{'peso':[random() for i in range(self.quantidade_camada_oculta)]} for i in range(self.quantidade_camada_saida)]
    #     self.rede.append(camada_saida)
    
    # def somar_net_n(self, entradas, camada_n):
    #     camada = self.rede[camada_n]

    #     for no in camada:
    #         soma = 0.00
    #         for peso, entrada in zip(no['peso'], entradas):
    #             soma += float(entrada * peso)
    #         no['net'] = soma   

    # def calcular_saida_camada_oculta_n(self, func_saida, camada_n):
    #     camada_oculta = self.rede[camada_n]
        
    #     for no in camada_oculta:
    #         saida = func_saida(no['net'])
    #         no['saida'] = saida

    # def calc_erro_saida(self, saidas_esperada, calc_saida_derivada):
    #     camada_saida = self.rede[-1]

    #     for no, saida_esperada in zip(camada_saida, saidas_esperada):
    #         erro_n = (saida_esperada - no['saida']) * calc_saida_derivada(no['saida'])
    #         no['erro'] = erro_n

    # def continuar_treinando(self):
    #     return True


    # def calc_erro_saida_n(self, camada_n):
    #     camada_atual = self.rede[camada_n]
    #     camada_proxima = self.rede[camada_n + 1]

    #     for index_camada_atual, no_atual in enumerate(camada_atual):
    #         erro_no_atual = 0.00
    #         for no_proximo in camada_proxima:
    #             erro_no_atual += float(no_proximo['peso'][index_camada_atual] * no_proximo['erro'] * self.funcao_saida_derivada(no_atual['net']))
    #         no_atual['erro'] = erro_no_atual


    # def calc_novos_pesos(self, entradas, camada_n):
    #     camada = self.rede[camada_n]
        
    #     for no in camada:
    #         for (index_peso, peso), entrada in zip(enumerate(no['peso']), entradas):
    #             novo_peso = peso + (self.taxa_aprendizagem * no['erro'] * entrada)
    #             no['peso'][index_peso] = novo_peso


    # def treinar_rede(self):

    #     for linha_x, linha in enumerate(self.grid_normalizada_hot_encoded):

    #         saida_esperada = self.classes_hot_encoded[linha_x]

    #         # soma net e saida da camada de saida
    #         self.somar_net_n(linha, camada_n=0)
    #         self.calcular_saida_camada_oculta_n(self.funcao_saida, camada_n=0)

    #         # pega a saida e que é a entrada do proximo
    #         saida_anterior = []
    #         for no in self.rede[0]:
    #             saida_anterior.append(no['saida'])

    #         # soma net e saida da camada de saida
    #         self.somar_net_n(saida_anterior, camada_n=1)
    #         self.calcular_saida_camada_oculta_n(self.funcao_saida, camada_n=1)

    #         # calcula o erro de saida
    #         self.calc_erro_saida(saida_esperada, self.funcao_saida_derivada)

    #         # verifica o erro geral da self.rede
    #         if not self.continuar_treinando():
    #             break

    #         # calcular erro camada oculta
    #         self.calc_erro_saida_n(camada_n=0)

    #         # calcular os novos pesos camada saida
    #         self.calc_novos_pesos(entradas=saida_anterior, camada_n=1)
            
    #         # calcular os novos pesos camada oculta
    #         self.calc_novos_pesos(entradas=linha, camada_n=0)

    #         pprint.pprint(self.rede, indent=4)


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
        pprint.pprint(self.grid_normalizada_hot_encoded)

    # def __quantidades_neuronios_camadas(self):
    #     self.quantidade_camada_entrada = len(self.grid_normalizada_hot_encoded[0])
    #     self.quantidade_camada_saida = len(self.cabecalho_grid_hot_encoded)
        
    #     calcular_quantidade_camada_oculta = lambda quantidade_camada_entrada, quantidade_camada_oculta: math.ceil((quantidade_camada_entrada + quantidade_camada_oculta) / 2)
        
    #     if self.quantidade_camada_oculta == 0:
    #         self.quantidade_camada_oculta = calcular_quantidade_camada_oculta(
    #             self.quantidade_camada_entrada, 
    #             self.quantidade_camada_oculta
    #         )
