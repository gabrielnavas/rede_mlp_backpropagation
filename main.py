from arquivo import Arquivo

from funcao_transferencia import FuncaoTransferencia
from configurar_rede_neural import ConfigurarDados
from rede_neural_testar import RedeNeuralTestar
from rede_neural_treinamento import RedeNeuralTreinamento

if __name__ == '__main__':

   # VARIAVEIS entrada
   funcao_transferencia_saida = FuncaoTransferencia.linear
   funcao_transferencia_derivada_saida = FuncaoTransferencia.linear_derivada
   # caminho_arquivo_treino = '/home/neo/Faculdade/TERMO_7/IA/2_BIM/TRABALHO/exemplo_base_dados/test.csv'
   caminho_arquivo_treinamento = '/home/neo/Faculdade/TERMO_7/IA/2_BIM/TRABALHO/exemplo_base_dados/base_treinamento.csv'
   caminho_arquivo_teste = '/home/neo/Faculdade/TERMO_7/IA/2_BIM/TRABALHO/exemplo_base_dados/base_test.csv'


   # ####### TREINAMENTO
   # configurar dados
   # recebe arquivo de treinamento de algum lugar
   dados = Arquivo(path=caminho_arquivo_treinamento)
   
   # le o arquivo
   dados.ler_arquivo()

   # torna a grid numerica, se der
   dados.tornar_itens_grid_numericos()
   
   # configura as classes (hot encoded etc) e os atributos
   rede_neural_configurada_treinamento = ConfigurarDados(
      grid=dados.grid, 
      colunas_grid_ativa=[False, True, True, False, True, True],
      cabecalho=dados.cabecalho, 
      classes=dados.classes,
   ) 
   rede_neural_configurada_treinamento.configurar()

   # rede neural para treinar
   rede_neural = RedeNeuralTreinamento(
      cabecalho=rede_neural_configurada_treinamento.cabecalho_grid_hot_encoded,
      classes=rede_neural_configurada_treinamento.classes_hot_encoded,
      grid=rede_neural_configurada_treinamento.grid_normalizada_hot_encoded,
      funcao_saida=funcao_transferencia_saida,
      funcao_saida_derivada=funcao_transferencia_derivada_saida,
      limiar_erro=0.00001,
      quantidade_iteracoes=2000,
      taxa_aprendizagem=1
   )

   # treinar a rede 
   rede_neural.treinar()
   print('acabo')

   # # ####### TESTE
   # # configurar dados
   # # recebe arquivo de treinamento de algum lugar
   # dados = Arquivo(path=caminho_arquivo_treinamento)
   
   # # le o arquivo
   # dados.ler_arquivo()

   # # torna a grid numerica, se der
   # dados.tornar_itens_grid_numericos()
   
   # # configura as classes (hot encoded etc) e os atributos
   # rede_neural_configurada_teste = ConfigurarDados(
   #    grid=dados.grid, 
   #    colunas_grid_ativa=[False, True, True, False, True, True],
   #    cabecalho=dados.cabecalho, 
   #    classes=dados.classes,
   # ) 
   # rede_neural_configurada_teste.configurar()


   # # testar rede
   # rede_neural.classes=rede_neural_configurada_teste.classes_hot_encoded,
   # rede_neural.grid=rede_neural_configurada_teste.grid_normalizada_hot_encoded,
   

   # # rede neural para treinar
   # rede_neural = RedeNeuralTestar(
   #    cabecalho=rede_neural_configurada_treinamento.cabecalho_grid_hot_encoded,
   #    classes=rede_neural_configurada_treinamento.classes_hot_encoded,
   #    grid=rede_neural_configurada_treinamento.grid_normalizada_hot_encoded,
   #    funcao_saida=funcao_transferencia_saida,
   #    funcao_saida_derivada=funcao_transferencia_derivada_saida,
   #    pesos_camada_oculta=rede_neural_configurada_treinamento.pesos_camada_oculta,
   #    pesos_camada_saida=rede_neural_configurada_treinamento.pesos_camada_saida,
   # )

   # rede_neural.testar()
   # matriz_confusao = rede_neural.gerar_matriz_confusao()
   # print(matriz_confusao)

   