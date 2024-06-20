import random
import matplotlib.pyplot as plt

# константы задачи
ONE_MAX_LENGTH = 100  # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
P_SELECTION = 0.6
MAX_GENERATIONS = 100  # максимальное количество поколений

RANDOM_SEED = 20
random.seed(RANDOM_SEED)


class FitnessMax():  # значение приспособленности особи
    def __init__(self):
        self.values = [0]


class Individual(list):  # представление каждой особи в популяции (список из 0 и 1)
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


def oneMaxFitness(individual):  #  определение функции принадлежности отдельной особи
    return sum(individual),  # кортеж


def individualCreator():  # функция для создания отдельного индивидуума
    return Individual([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])


def populationCreator(n=0):  # функции для создания популяции
    return list([individualCreator() for i in range(n)])


population = populationCreator(n=POPULATION_SIZE)  # создаем начальную популяцию
generationCounter = 0  # счетчик поколений

fitnessValues = list(map(oneMaxFitness,
                         population))  # Вычисляем текущие значения приспособленности для каждой особи в начальной популяции и сохраняем эти значения в свойстве values каждого индивидуума

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []  # вспомогательный список для хранения лучшей приспособленности в каждом текущем поколении
meanFitnessValues = []  # вспомогательный список для хранения средней приспособленности в каждом текущем поколении


# Функции для клонирования индивида, выполнения турнирного отбора, одноточечного скрещивания и мутации
def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind


def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))
        if random.random() < P_SELECTION:
            offspring.append(
                max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring


def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1) - 3)
    child1[s:], child2[s:] = child2[s:], child1[s:]


def mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1


# вычислим список значений приспособленностей для всех хромосом в популяции
fitnessValues = [individual.fitness.values[0] for individual in population]

while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))

    fitnessValuess = [ind.fitness.values[0] for ind in
                      offspring]
    counter = len(offspring) - 200
    for i in range(0, counter):
        minI = fitnessValuess.index(min(fitnessValuess))
        del fitnessValuess[minI]
        del offspring[minI]

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant, indpb=1.0 / ONE_MAX_LENGTH)

    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness.values[0] for ind in population]

    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")

    for s in range(0, 10):
        index = random.randint(0, 189)
        del population[index]

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
