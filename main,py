import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define the neural network architecture
def create_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Define the fitness function
def fitness_function(population, x_train, y_train, x_test, y_test):
    fitness_scores = []
    for individual in population:
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.set_weights(individual)
        loss, accuracy = model.evaluate(x_test, y_test)
        fitness_scores.append(accuracy)
    return fitness_scores

# Initialize the population
population_size = 10
population = [create_model().get_weights() for i in range(population_size)]

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Implement the genetic algorithm
num_generations = 25
mutation_rate = 0.91413531535

for i in range(num_generations):
    fitness_scores = fitness_function(population, x_train, y_train, x_test, y_test)
    fittest_individual = population[np.argmax(fitness_scores)]
    new_population = [fittest_individual]
    while len(new_population) < population_size:
        parent1 = population[np.random.choice(len(population), p=fitness_scores/np.sum(fitness_scores))]
        parent2 = population[np.random.choice(len(population), p=fitness_scores/np.sum(fitness_scores))]
        child = []
        for i in range(len(parent1)):
            if np.random.uniform() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        for i in range(len(child)):
            if np.random.uniform() < mutation_rate:
                child[i] += np.random.normal(0, 0.1, child[i].shape)
        new_population.append(child)
    population = new_population

# Train the neural network
model = create_model()
model.set_weights(fittest_individual)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
