import numpy as np
import math
import random


weights = dict()

population_size = 10
#pacman_population = [pacman(0, 0) for _ in range(population_size)]
n_generations = 10000000

# initializing weights for neural net for 10 members of population
'''
layer_1 = np.zeros((27, 52))  # L x (L-1)
#np.random.uniform(layer_1)
layer_2 = np.zeros((52, 26))
#np.random.uniform(layer_2)
layer_3 = np.zeros((26, 13))    #((capa entrada, capa salida))
#np.random.uniform(layer_3)
layer_4 = np.zeros((13, 4))
#np.random.uniform(layer_4)
'''
layer_1 = np.zeros((12, 32))  # Primera capa con 14 nodos de entrada
layer_2 = np.zeros((32, 16))  # Segunda capa con 20 nodos
layer_3 = np.zeros((16, 8))   # Tercera capa con 14 nodos
layer_4 = np.zeros((8, 4))    # Cuarta capa con 4 nodos de salida

for i in range(population_size):
    weights[i] = [np.random.randn(*layer_1.shape) * np.sqrt(1 / layer_1.shape[0]),  # Xavier para capa 1
        np.random.randn(*layer_2.shape) * np.sqrt(1 / layer_2.shape[0]),  # y para las siguientes
        np.random.randn(*layer_3.shape) * np.sqrt(1 / layer_3.shape[0]),
        np.random.randn(*layer_4.shape) * np.sqrt(1 / layer_4.shape[0])]


def softmax(output_layer):
    sum_e = 0
    # print(f"output layer full {output_layer}")            #convierte la salida en probabilidad
    for i in range(len(output_layer)):
        # sum_e += math.e**(output_layer[:, i])
        sum_e += math.e**(output_layer[i])
    return math.e**(output_layer)/sum_e

def sigmoid(x):                             #funcion de activacion para regular senales entre capas. mantiene valor entre 0 y 1

    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def neural_net(sample, input_ga): #no se usa


    temp2 = relu(np.matmul(input_ga.T, weights[sample][0]))          #multiplica matrices T traspuesta
    temp3 = relu(np.matmul(temp2, weights[sample][1]))           #sample es el indice en el array y el segundo[] es la capa de x individuo
    temp4 = relu(np.matmul(temp3, weights[sample][2]))
    out = softmax(np.matmul(temp4, weights[sample][3]))         #convertimos en la probabilidad final

    # print(f"i am out {out}")
    ind = np.argmax(out)                #printea el valor mas alto de la matriz de salida.
    return ind


def cal_pop_fitness(score, time, ghosDistance):
    #close_ghosts = 0
    fitness = 0
    return score/1960
    #for distance in ghosDistance:
    #    if distance <0.6:
    #        close_ghosts += 1

    #fitness = fitness + score

    #if close_ghosts == 1:
    #    fitness *=0.9 * time
    #elif close_ghosts == 2:
    #    fitness *=0.7 * time
    #elif close_ghosts == 3:
    #    fitness *=0.5
    #elif close_ghosts == 4:
    #    fitness *=0.2
    #if fitness < 0:
    #    return 0
    #else:
    #    return fitness




'''def mutation(offspring_crossover, percent_mutation):
    # Mutar todos los elementos en la matriz de forma aleatoria
    num_rows, num_cols = offspring_crossover.shape
    variance = 2 / (offspring_crossover.shape[0] + offspring_crossover.shape[1])
    std_dev = np.sqrt(variance)
    # Iterar sobre cada elemento de la matriz
    for i in range(num_rows):
        for j in range(num_cols):
            # Aplicar mutación al gen con una probabilidad percent_mutation
            if random.random() < percent_mutation:
                offspring_crossover[i, j] = np.random.normal(0, std_dev)

    return offspring_crossover'''
def mutation(offspring_crossover, percent_mutation):
    # Calcular la varianza y la desviación estándar para la mutación suave
    num_rows, num_cols = offspring_crossover.shape
    variance = 2 / (num_rows + num_cols)
    std_dev = np.sqrt(variance)

    # Aplicar elitismo (probabilidad de evitar mutación completa en este individuo)
    if np.random.rand() > 0.1:  # 90% de probabilidad de aplicar mutación
        # Mutación suave en una fracción de los genes, seleccionados aleatoriamente
        for i in range(num_rows):
            for j in range(num_cols):
                # Aplicar mutación suave con probabilidad percent_mutation
                if np.random.rand() < percent_mutation:
                    offspring_crossover[i, j] += np.random.normal(0, std_dev)

    # Aplicar mutación de multiplicación proporcional en todos los genes
    multiplier = np.random.uniform(0.95, 1.05, size=(num_rows, num_cols))
    offspring_crossover *= multiplier

    return offspring_crossover


def mutate(weights, mutation_rate, mutation_strength=0.1):
    # Aplicar elitismo
    if np.random.rand() > 0.1:  # 90% de probabilidad de mutar
        for layer in weights:
            # Realizar mutación suave en una fracción de los pesos
            mask = np.random.rand(*layer.shape) < mutation_rate
            layer += mask * np.random.normal(0, mutation_strength, size=layer.shape)

    # Aplicar multiplicación en vez de adición para un ajuste proporcional
    for layer in weights:
        layer *= np.random.uniform(0.95, 1.05, size=layer.shape)

    return weights


def crossover(mating_pool):          #indices en la poblacion que haran de padres
    global weights
    global population_size  # n_generation
    new_children_weights = dict()  # genes    #inicializo el nuevo hijo
    for child in range(population_size):  # n_generation
        child_net = []
        p1 = random.choice(mating_pool)
        p2 = random.choice(mating_pool)
        while np.array_equal(p1, p2):
            p2 = random.choice(mating_pool)
        for layer in range(4):
            p1Matrix = p1[layer]  # los pesos de los padres de la capa aactual en matriz lo levanto
            p2Matrix = p2[layer]

            crossed = ((p1Matrix + p2Matrix) / 2)
            crossed = mutation(crossed, 0.01)

            child_net.append(crossed)
            # child_net.append(crossed.reshape(p1Matrix.shape[0], p1Matrix.shape[1]))      #lo volvemos a pasar como matriz
        new_children_weights[child] = child_net
    weights = new_children_weights