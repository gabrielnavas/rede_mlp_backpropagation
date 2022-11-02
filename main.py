from random import seed
from random import random

import pprint
from arquivo import Arquivo

from funcao_transferencia import FuncaoTransferencia
from rede_neural import RedeNeural

# def iniciar_rede(bias, n_inputs, n_hidden, n_outputs):
#    seed(1)

#    rede = list()
#    bias = 1 if bias else 0
   
#    camada_oculta = [{'peso':[random() for i in range(n_inputs + bias)]} for i in range(n_hidden)]
#    rede.append(camada_oculta)
   
#    camada_saida = [{'peso':[random() for i in range(n_hidden + bias)]} for i in range(n_outputs)]
#    rede.append(camada_saida)
   
#    return rede

# def somar_net_n(rede, entradas, camada_n):
#    camada = rede[camada_n]

#    for no in camada:
#       soma = 0.00
#       for peso, entrada in zip(no['peso'], entradas):
#          soma += float(entrada * peso)
#       no['net'] = soma   
   
#    return rede

# def calcular_saida_camada_oculta_n(rede, func_saida, camada_n):
#    camada_oculta = rede[camada_n]
   
#    for no in camada_oculta:
#       saida = func_saida(no['net'])
#       no['saida'] = saida

#    return rede


# def calc_erro_saida(rede, saidas_esperada, calc_saida_derivada):
#    camada_saida = rede[-1]

#    for no, saida_esperada in zip(camada_saida, saidas_esperada):
#       erro_n = (saida_esperada - no['saida']) * calc_saida_derivada(no['saida'])
#       no['erro'] = erro_n

#    return rede

# def continuar_treinando(rede):
#    return True

# def calc_erro_saida_n(rede, camada_n, calc_saida_derivada):
#    camada_atual = rede[camada_n]
#    camada_proxima = rede[camada_n + 1]

#    for index_camada_atual, no_atual in enumerate(camada_atual):
#       erro_no_atual = 0.00
#       for no_proximo in camada_proxima:
#          erro_no_atual += float(no_proximo['peso'][index_camada_atual] * no_proximo['erro'] * calc_saida_derivada(no_atual['net']))
#       no_atual['erro'] = erro_no_atual

#    return rede

# def calc_novos_pesos(rede, taxa_aprendizagem, entradas, camada_n):
#    camada = rede[camada_n]
   
#    for no in camada:
#       for (index_peso, peso), entrada in zip(enumerate(no['peso']), entradas):
#          novo_peso = peso + (taxa_aprendizagem * no['erro'] * entrada)
#          no['peso'][index_peso] = novo_peso

#    return rede

# def treinar_rede(rede, dados, taxa_aprendizagem):
#    taxa_aprendizagem = 1

#    for linha in dados:

#       saida_esperada = linha[-1]
#       linha = linha[0:-1]

#       # soma net e saida da camada de saida
#       rede = somar_net_n(rede, linha, camada_n=0)
#       rede = calcular_saida_camada_oculta_n(rede, FuncaoTransferencia().linear, camada_n=0)

#       # pega a saida e que é a entrada do proximo
#       saida_anterior = []
#       for no in rede[0]:
#          saida_anterior.append(no['saida'])

#       # soma net e saida da camada de saida
#       rede = somar_net_n(rede, saida_anterior, camada_n=1)
#       rede = calcular_saida_camada_oculta_n(rede, FuncaoTransferencia().linear, camada_n=1)

#       # calcula o erro de saida
#       rede = calc_erro_saida(rede, saida_esperada, FuncaoTransferencia().linear_derivada)

#       # verifica o erro geral da rede
#       if not continuar_treinando(rede):
#          break

#       # calcular erro camada oculta
#       rede = calc_erro_saida_n(rede, camada_n=0)

#       # calcular os novos pesos camada saida
#       rede = calc_novos_pesos(rede, taxa_aprendizagem=taxa_aprendizagem, entradas=saida_anterior, camada_n=1)
      
#       # calcular os novos pesos camada oculta
#       rede = calc_novos_pesos(rede, taxa_aprendizagem=taxa_aprendizagem, entradas=linha, camada_n=0)

#       pprint.pprint(rede, indent=4)

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
      classes=dados.classes,
      funcao_saida=FuncaoTransferencia.linear,
      funcao_saida_derivada=FuncaoTransferencia.linear_derivada,
      limiar_erro=0.00001,
      quantidade_iteracoes=2000,
      taxa_aprendizagem=1
   ) 

   # configura as classes (hot encoded etc) e os atributos
   rede.configurar()

   # iniciar rede
   rede.iniciar_rede()

   # treinar a rede 
   # TODO: essa funcao não está funcionando
   rede.treinar_rede()

   pprint.pprint(rede.cabecalho_grid_hot_encoded)
   # pprint.pprint(rede.classes_hot_encoded)
   