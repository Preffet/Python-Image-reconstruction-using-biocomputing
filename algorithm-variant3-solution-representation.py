#!/usr/bin/env python3

import multiprocessing
import random
import json
import sys
import matplotlib.pyplot as plt
import numpy
import shapely
import numpy as np
from deap import creator, base, tools, algorithms
from PIL import Image, ImageDraw, ImageChops
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

x_points = np.array([])
y_points = np.array([])
number_of_polygons = 100

# ANSI escape codes to print coloured text
class colours:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ORANGE = '\033[38;5;173m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


# read parameters from JSON configuration files
def read_parameters_from_file(script_file_name):
    global number_of_generations
    global population_size
    global cx_probability   # probability of mating two individuals
    global mut_probability  # probability for mutating an individual
    global target_image
    try:
        f = open(script_file_name)
        data = json.load(f)
        f.close()
    except:
        print(f"{colours.BOLD}There was an error reading parameters from"
              f" the {colours.ENDC}{colours.RED}{script_file_name}{colours.ENDC}"
              f"{colours.BOLD} file{colours.ENDC}")
        sys.exit()
    number_of_generations = data['numberOfGenerations']
    population_size = data['populationSize']
    cx_probability = data['cxProbability']
    mut_probability = data['mutProbability']
    target_image = data['targetImage']

    print(f"\n{colours.BOLD}The script will be run with following parameters:\n"
          f"-------------------------------------------------{colours.ENDC}")
    print(f"{'Number of generations':<45}{colours.BOLD}{colours.BLUE}{number_of_generations}{colours.ENDC}")
    print(f"{'Population size':<45}{colours.BOLD}{colours.CYAN}{population_size}{colours.ENDC}")
    print(f"{'Probability of mating two individuals':<45}{colours.BOLD}{colours.GREEN}{cx_probability}{colours.ENDC} ")
    print(
        f"{'Probability for mutating an individual:':<45}{colours.BOLD}{colours.YELLOW}{mut_probability}{colours.ENDC}")
    print(f"{'Target image:':<45}{colours.BOLD}{colours.ORANGE}{target_image}{colours.ENDC}\n")


# ask if user would like to run the script with new custom parameters,
# read them from config.JSON file or re-run with previous parameters.
# Also, process the parameters and save them to previousConfig.JSON
# file so that the user could easily re-run the script using them.
def process_parameters():
    global number_of_generations
    global population_size
    global cx_probability   # probability of mating two individuals
    global mut_probability  # probability for mutating an individual
    global target_image
    # ask user if parameters should be read from file
    yes = {'yes', 'y'}
    no = {'no', 'n'}  # pylint: disable=invalid-name
    done = False
    run_with_previous_cfg = False
    read_from_file = False

    print(f"\n{colours.BOLD}Read the parameters from the config.json file?{colours.ENDC} "
          f"({colours.GREEN}y{colours.ENDC}/{colours.RED}n{colours.ENDC})")
    while not done:
        choice = input().lower()
        if choice in yes:
            read_from_file = True
            done = True
        elif choice in no:
            read_from_file = False
            done = True
        else:
            print(f"{colours.BOLD}Please respond by yes or no {colours.ENDC}"
                  f"({colours.GREEN}y{colours.ENDC}/{colours.RED}n{colours.ENDC})")
    done = False
    # read parameters from config file
    if read_from_file:
        read_parameters_from_file('config.JSON')
    # ask if the script should be re-run with previous configuration
    else:
        print(f"\n{colours.BOLD}Run the script with previous configuration?{colours.ENDC} "
              f"({colours.GREEN}y{colours.ENDC}/{colours.RED}n{colours.ENDC})")
        while not done:
            choice = input().lower()
            if choice in yes:
                run_with_previous_cfg = True
                done = True
            elif choice in no:
                run_with_previous_cfg = False
                done = True
            else:
                print(f"{colours.BOLD}Please respond by yes or no {colours.ENDC}"
                      f"({colours.GREEN}y{colours.ENDC}/{colours.RED}n{colours.ENDC})")
        if run_with_previous_cfg:
            read_parameters_from_file('previousConfig.JSON')

    # get parameters from manual input
    if (not read_from_file) and (not run_with_previous_cfg):
        valid_input = False
        while not valid_input:
            try:
                print(f"\n{colours.BOLD}Please enter the configuration manually\n"
                      f"{colours.BLUE}------------{colours.CYAN}------------{colours.GREEN}------------"
                      f"{colours.YELLOW}------------{colours.ENDC}")
                number_of_generations = int(input(f"{colours.BOLD}{colours.BLUE}~>"
                                                  f" {colours.ENDC}"f"{'Number of generations':<41}"))
                population_size = int(input(f"{colours.BOLD}{colours.CYAN}~> {colours.ENDC}{'Population size':<41}"))
                cx_probability = float(input(
                    f"{colours.BOLD}{colours.GREEN}~> {colours.ENDC}{'Probability of mating two individuals':<41}"))
                mut_probability = float(input(
                    f"{colours.BOLD}{colours.YELLOW}~> {colours.ENDC}{'Probability for mutating an individual':<41}"))
                target_image = input(
                    f"{colours.BOLD}{colours.ORANGE}~> {colours.ENDC}{'Target image':<41}")
                Image.open(target_image)
                print("\n")
                valid_input = True

            except (ValueError, IOError):
                print(f"\n{colours.BOLD}{colours.RED}Provided parameters are invalid. Please try again. {colours.ENDC}")

            # check if given arguments are valid
            else:
                if mut_probability < 0 or cx_probability < 0 or cx_probability > 1 or \
                        mut_probability > 1 or number_of_generations < 2 or population_size < 1:
                    print(
                        f"{colours.BOLD}{colours.RED}Provided parameters are invalid. Please try again. {colours.ENDC}")
                    valid_input = False

    # save the configuration to previousConfig.JSON file so that the user
    # could easily run the script again using the same configuration
    configuration_dict = {
        "numberOfGenerations": number_of_generations,
        "populationSize": population_size,
        "cxProbability": cx_probability,
        "mutProbability": mut_probability,
        "targetImage": target_image
    }
    with open('previousConfig.JSON', 'w', encoding='utf-8') as f:
        json.dump(configuration_dict, f, ensure_ascii=False, indent=4)


# draw the polygons
def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    image.save("solution.png")
    return image


# evaluate the solution
def evaluate(solution):

    target = Image.open(target_image)
    target.load()  # read image and close the file
    maximum = 255 * target.size[0] * target.size[1]

    image = draw(solution)
    diff = ImageChops.difference(image, target)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (maximum - count) / maximum,


def make_polygon():
    # generate valid polygons (pentagons)
    polygon_area = 0
    not_self_intersecting = False
    while not not_self_intersecting and polygon_area < 20:
        x1 = random.randrange(10, 190)
        y1 = random.randrange(10, 190)

        x2 = random.randrange(10, 190)
        y2 = random.randrange(10, 190)

        x3 = random.randrange(10, 190)
        y3 = random.randrange(10, 190)

        x4 = random.randrange(10, 190)
        y4 = random.randrange(10, 190)

        x5 = random.randrange(10, 190)
        y5 = random.randrange(10, 190)

        # check polygons for self-intersection
        polygon_coordinates = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
        pentagon = shapely.geometry.polygon.Polygon(polygon_coordinates)
        if pentagon.is_valid:
            not_self_intersecting = False
        else:
            not_self_intersecting = True
        polygon_area = pentagon.area

    # return valid polygons
    return ([(random.randrange(0, 256), random.randrange(0, 256),
              random.randrange(0, 256), random.randrange(30, 61)),
             (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)])


# re-order polygons or mutate either coordinates or RGBA values
def mutate(solution, indpb):
    global balanced_mut_probability
    k = random.randint(0, 1)  # 1 - mutate colour and opacity, 0 - mutate coordinates
    if random.random() < balanced_mut_probability:
        # mutate 2% of the points
        for i in range(int(number_of_polygons*0.02)):
            polygon = random.choice(solution)
            coords = [x for point in polygon[1:] for x in point]
            polygon_colours = list(polygon[0][:3])
            opacity = [(polygon[0][3])]

            # mutate points
            if k == 1:
                tools.mutGaussian(coords, 0, 10, indpb)
                coords = [max(0, min(int(x), 200)) for x in coords]
                polygon[1:] = list(zip(coords[::2], coords[1::2]))

            # change colour and opacity
            if k == 0:
                tools.mutGaussian(polygon_colours, 0, 10, indpb)

                tools.mutGaussian(opacity, 0, 10, indpb)
                opacity = [max(0, min(int(x), 60)) for x in opacity]

                polygon_colours = [max(0, min(int(x), 255)) for x in polygon_colours]
                polygon[0] = tuple(polygon_colours + opacity)

    else:
        # perform a more drastic mutation to RGBA values
        # (3% of the polygons)
        if k == 1:
            for i in range(int(number_of_polygons * 0.03)):
                polygon = random.choice(solution)
                polygon_colours = list(polygon[0][:3])
                opacity = [(polygon[0][3])]
                tools.mutGaussian(polygon_colours, 0, 20, indpb)

                tools.mutGaussian(opacity, 0, 20, indpb)
                opacity = [max(0, min(int(x), 60)) for x in opacity]

                polygon_colours = [max(0, min(int(x), 255)) for x in polygon_colours]
                polygon[0] = tuple(polygon_colours + opacity)
        # or reorder the polygons
        else:
            tools.mutShuffleIndexes(solution, indpb)
    return solution,


def run():
    global x_points, y_points, mut_probability, balanced_mut_probability
    toolbox = base.Toolbox()
    pool = multiprocessing.Pool(8)
    toolbox.register("map", pool.map)
    toolbox.register("individual", tools.initRepeat, creator.Individual, make_polygon, n=100)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)
    # Register the mutation operator
    toolbox.register("mutate", mutate, indpb=mut_probability)
    # Register the tournament selection operator
    toolbox.register("select", tools.selTournament, tournsize=100)
    # Register the evaluation operator
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=population_size)

    # Evaluate the entire population
    fitness_values = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitness_values):
        ind.fitness.values = fit

    # Extract all the fitness values
    fits = [ind.fitness.values[0] for ind in population]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution (main evolution loop)
    while g < number_of_generations and round(max(fits), 5) < 0.95:

        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation of individuals
        # 10% elitism 90% tournament selection
        offspring = tools.selBest(population,
                                  int(0.1 * len(population))) + toolbox.select(population, len(population) - int(
            0.1 * len(population)))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # calculate the mean fitness value
        fits_before_cx = [ind.fitness.values[0] for ind in offspring]
        mean_before_cx = sum(fits_before_cx) / len(offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # only use crossover if both children have lower fitness values
            # than average
            if (child1.fitness.values[0] != child2.fitness.values[0] and
                    ((child1.fitness.values[0] < mean_before_cx) or
                     child2.fitness.values[0] < mean_before_cx)):

                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        balanced_mut_probability = mut_probability
        # Apply mutation on the offspring
        for mutant in offspring:
            mutant_ind = [ind for ind in mutant]
            mutant_fitness = evaluate(mutant_ind)
            # decrease chance to mutate points slowly if fitness is low
            if (mutant_fitness[0] < mean_before_cx):
                balanced_mut_probability = mut_probability - mut_probability*0.2
            if random.random() < mut_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            # reset mut probability back to original
            balanced_mut_probability = mut_probability


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness_values = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness_values):
            ind.fitness.values = fit

        # replace the old population with new individuals
        population[:] = offspring

        # Gather all fitness values in one list
        fits = [ind.fitness.values[0] for ind in population]

        # calculate statistics
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        # every 100 generations save the statistics to
        # x_points and y_points variables for diagram plotting
        if g % 100 == 0 or round(max(fits), 4) == 0.95:
            x_points = numpy.append(x_points, g)
            y_points = numpy.append(y_points, round(max(fits), 3))

        # print the statistics
        print(f"|  MIN {colours.RED}{ round(min(fits), 4)}{colours.ENDC}  |  MAX {colours.GREEN}{round(max(fits), 4)}{colours.ENDC}"
              f"  |  AVG {round(mean, 4)}  |  STD {round(std, 4)} \n")


if __name__ == "__main__":
    process_parameters()
    run()
    # after running the evolution, plot the
    # graph which illustrates how maximum fittness value
    # changed every 100 generations
    points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots()
    graph_title = 'Solution representation algorithm ' + target_image
    ax.set_title(graph_title)
    ax.set_ylabel('Max fitness')
    ax.set_xlabel('Number of generations')
    lc = LineCollection(segments, cmap='cool')
    lc.set_array(y_points)
    lc.set_linewidth(5)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    plt.plot(x_points, y_points, linewidth=0.1)
    plt.show()
