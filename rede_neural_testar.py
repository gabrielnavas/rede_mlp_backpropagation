import numpy as np
from typing import List

class RedeNeuralTestar:
    def __init__(
        self, 
        grid: List[List[float]], 
        classes: List[list[float]], 
        cabecalho: List[str], 
        pesos_camada_oculta: List[List[float]],
        pesos_camada_saida: List[List[float]],
        funcao_saida,
        funcao_saida_derivada,
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

        # funções de calculo apos a somatoria nos neurónios
        self.funcao_saida = funcao_saida    
        self.funcao_saida_derivada = funcao_saida_derivada
 
        # pesos da rede
        self.pesos_camada_oculta = pesos_camada_oculta
        self.pesos_camada_saida = pesos_camada_saida


    def testar(self):
        # iniciar numero de epocas
        self.epocas = 0

        saidas_camada_oculta = []
        saidas_camada_saida = []

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

    
    def gerar_matriz_confusao(self):
        return []