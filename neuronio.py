import math
import random
# a bibliotecas acimas foram usadas unicamente para gerar RUs aleatoreamente

meu_ru = 3935883 
entradas = [meu_ru] # lista que armazena RUs geradas
gerar = 19 # optei por usar 19 rus aleatorias, totalizando 20 entradas para o perceptron

def perceptron():

    while True:
        try:
            random.seed(10) # parametro impede a geracao de novos valores em cada execucao

            # gerar RUs aleatoreamente, acima e abaixo do valor da minha RU
            for i in range(0, gerar):
                gerar_ru = math.ceil(random.triangular(3000000, 5000000)) # gerar ru e arrendondar os valores float
                entradas.append(gerar_ru)

            pesos = []
            bias = 0 # o vies escolhido por padrao foi 0
            
            for i in entradas:
                pesos.append(1) # o peso escolhido por padrao foi 1

            input_function = list(map(lambda a, b: a * b, entradas, pesos))
            
            print(entradas)
            print(pesos)
            print(input_function)

        except ValueError:
            print('falha no processo')

perceptron()