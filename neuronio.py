#import math
#import random
import time # marcar tempo de execucao

# ------ parametros inicias --------------------------

# ENTRADAS
meu_ru = 3935883 
rus = [meu_ru, 4074152, 3926164, 4081405, 3642026, 4388971, 4406012, 4167501, 3566092, 4020888,
         3809658, 3707103, 4692810, 4917018, 3298518, 4471155, 4109148, 3873621, 3753152, 4193731] # lista que armazena RUs geradas, comecando ja com meu RU

# O bloco de codigo abaixo foi utilizado para gerar os RUs na lista acima
"""random.seed(10) # parametro impede a geracao de novos valores em cada execucao
for i in range(0, gerar):  # gerar RUs aleatoreamente, acima e abaixo do valor da minha RU
    gerar_ru = math.ceil(random.triangular(3000000, 5000000)) # gerar ru e arrendondar os valores float
    rus.append(gerar_ru)
"""
# PESOS
peso_inicial = 1    # o peso escolhido por padrao foi 1

# VIES
bias = 0 # o vies escolhido por padrao foi 0

# ATIVADORES (tanh)
a = 1
b = -1

# TAXA DE APRENDIZADO
k = 0.01

# classificacao humana
classificacao = [1,1,-1,1,-1,1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,1]



# ------ Perceptron --------------------------------

def perceptron(dados, peso_inicial, bias, alvos, taxa):

    inicio = time.time() # marca inicio
    k = taxa
    passo = 0 # contador de passos do loop de aprendizado

    entradas = {}
    for c, dado in zip(range(len(dados)), dados):
        entrada = []
        for d in str(dado):
            entrada.append(int(d))
        entradas[c+1] = entrada

    print('entradas {}'.format(entradas))

    pesos = []
    
    for c in range(len(str(meu_ru))):
        pesos.append(peso_inicial)
    
    print('pesos {}'.format(pesos))

    print('RUs: {} \n (o primeiro e o meu!) \n Alvos: {} \n Pesos iniciais: {}'.format(dados, alvos, pesos))

    # loop de aprendizado
    while True:
        try:
            def input_function(entradas, pesos, bias):
                
                # calcular saida das funcoes
                saidas = {}
                for chave, entrada in entradas.items():
                    soma = []
                    for x, w in zip(entrada, pesos):
                        s = x * w + bias # funcao input
                        soma.append(s) 

                    saidas[chave] = sum(soma) # somar e salvar no dicionario

                return saidas

            saidas = input_function(entradas, pesos, bias)

            def funcao_ativador(a, b, saidas):
                
                atuais = {} # classificacoes do perceptron
                for chave, saida in saidas.items():
                    # ativacao
                    if saida >= 0:
                        atuais[chave] = a
                    else:
                        atuais[chave] = b

                return atuais

            atuais = funcao_ativador(a, b, saidas)

            def retro_propagacao(entradas, k, atuais, alvos, pesos): # baseado na regra de Widrow-Hoff (Regra Delta)
                
                # calcular e acumular erros
                erros = {}
                for chave, atual, alvo in zip(atuais.keys(), atuais.values(), alvos):
                    erro = atual - alvo
                    erros[chave] = erro
                
                # calcular e acumular deltas
                deltas_totais = {}
                for chave, erro, entrada in zip(erros.keys(), erros.values(), entradas.values()):
                    deltas = []
                    for x in entrada:
                        delta = erro * x * k
                        deltas.append(delta)                        
                    deltas_totais[chave] = deltas

                # calcular media de dos deltas
                medias = []
                for chave, delta in zip(deltas_totais.keys(), deltas_totais.values()):
                    media_deltas = round(sum(delta)/len(delta), 2)
                    medias.append(media_deltas)
                
                # calcular pesos ajustados
                pesos_ajustados = []
                for peso, media in zip(pesos, medias):
                    pesos_ajustados.append(round(peso - media, 2))
                
                # ajustar pesos
                pesos.clear()
                for peso_ajustado in pesos_ajustados:
                    pesos.append(peso_ajustado)
                    
                return erros, deltas_totais

            erros, deltas = retro_propagacao(entradas, k, atuais, alvos, pesos)
            
            # OUTPUT
            passo += 1
    
            print('PASSO {}'.format(passo))
            print('PESOS: {}'.format(pesos))

            for chave, valor, saida, atual, alvo, erro, delta in zip(entradas.keys(), entradas.values(), 
                    saidas.values(), atuais.values(), alvos, erros.values(), deltas.values()):
                print('Entrada {}: {} | saida: {} | atual: {} | alvo: {}| erro: {} | deltas: {}'
                        .format(chave, valor, round(saida, 2), atual, alvo, erro, delta))
            
            somar_erros = []
            for erro in erros.values():
                somar_erros.append(erro)

            print('SOMA DOS ERROS: {}'.format(sum(somar_erros)))

            # interromper o aprendizado quando todos a soma dos erros zerar
            if sum(somar_erros) == 0:
                break

            else:
                pass

        except ValueError:
            print('falha no processo')
            break
        
        # performance
        finally:
            fim = time.time()
            print('TOTAL DE PASSOS: {} | TEMPO: {} s'.format(passo, fim - inicio))

perceptron(rus, peso_inicial, bias, classificacao, k)