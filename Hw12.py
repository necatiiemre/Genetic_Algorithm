import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# Read Image
def read_image():
    img_gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    return img_gray

# Divide Image
def slice_image(img, num_pieces):
    height, width = img.shape
    aspect_ratio = width / height

    # Find a piece size that is close to the aspect ratio
    factors = []
    for i in range(1, int(math.sqrt(num_pieces)) + 1):
        if num_pieces % i == 0:
            factors.append((i, num_pieces // i))

    # Find the closest piece size
    best_pair = None
    min_diff = float('inf')
    for rows, cols in factors:
        current_ratio = cols / rows
        diff = abs(current_ratio - aspect_ratio)
        if diff < min_diff:
            min_diff = diff
            best_pair = (rows, cols)

    if best_pair is None:
        # If no suitable pair is found, use a single row
        rows, cols = 1, num_pieces
    else:
        rows, cols = best_pair

    piece_width = width // cols
    piece_height = height // rows
    pieces = []

    for row in range(rows):
        for col in range(cols):
            left = col * piece_width
            upper = row * piece_height
            right = left + piece_width
            lower = upper + piece_height
            piece = img[upper:lower, left:right]
            pieces.append(piece)

    return pieces, (piece_height, piece_width), (rows, cols)

# Piece Shuffle
def shuffle_pieces(pieces):
    shuffled_pieces = pieces.copy()
    random.shuffle(shuffled_pieces)
    return shuffled_pieces

# Reconstruct image
def reconstruct_image(pieces, piece_size, rows, cols):
    piece_height, piece_width = piece_size
    reconstructed_image = np.zeros((piece_height * rows, piece_width * cols), dtype=pieces[0].dtype)

    index = 0
    for row in range(rows):
        for col in range(cols):
            upper = row * piece_height
            left = col * piece_width
            reconstructed_image[upper:upper + piece_height, left:left + piece_width] = pieces[index]
            index += 1

    return reconstructed_image

# Fitness Function
def fitness_function(individual, original_pieces):
    correct_positions = sum(1 for i in range(len(individual)) if np.array_equal(individual[i], original_pieces[i]))
    return correct_positions

# Find Parents
def select_parents(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[-2:]  # En iyi iki bireyi seç
    return [population[selected_indices[0]], population[selected_indices[1]]]

# Crossover Funtcion
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    
    def is_piece_in_list(piece, pieces_list):
        return any(np.array_equal(piece, p) for p in pieces_list)
    
    child = parent1[:cut] + [piece for piece in parent2 if not is_piece_in_list(piece, parent1[:cut])]
    return child

# Mutation Function
def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Fitness Scores Plot
def plot_fitness_scores(generation, fitness_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(fitness_scores)), fitness_scores, color='blue')
    plt.xlabel('Birey')
    plt.ylabel('Fitness Skoru')
    plt.ylim(0, max(fitness_scores) + 1)
    plt.title(f'Generation {generation} Fitness Scores')
    
    # X axis ticks
    plt.xticks(range(len(fitness_scores)), range(1, len(fitness_scores) + 1))
    
    plt.savefig(f"output_graph/Generation_{generation}_fitness_scores.png")
    plt.close()

# Genetic Algorithm
def genetic_algorithm(shuffled_pieces, original_pieces, piece_size, rows, cols, population_size=20, generations=1000):
    population = [shuffle_pieces(shuffled_pieces) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [fitness_function(individual, original_pieces) for individual in population]
        
        # Fitness scores plot save
        plot_fitness_scores(generation, fitness_scores)
        
        # Print fitness scores for the current generation
        print(f"Generation {generation}: Fitness Scores = {fitness_scores}")
        
        if max(fitness_scores) == len(original_pieces):  # Tam doğru birey bulunduysa
            best_individual = population[np.argmax(fitness_scores)]
            print(f"Solved in generation {generation}")
            return best_individual

        parents = select_parents(population, fitness_scores)
        
        # Generation Best Individual save and show
        population_best_individual = reconstruct_image(parents[1], piece_size, rows, cols)
        cv2.imshow(f"Generation {generation} Best Individual", population_best_individual)
        cv2.imwrite(f"output_images/Generation_{generation}_best_individual.png", population_best_individual)
        cv2.waitKey(1)
        
        next_generation = []

        for _ in range(population_size):
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            next_generation.append(child)

        population = next_generation

    best_individual = population[np.argmax([fitness_function(ind, original_pieces) for ind in population])]
    return best_individual

# Preprocessing
def preprocessing(patch_size):
    img = read_image()
    pieces, piece_size, (rows, cols) = slice_image(img, patch_size)
    shuffled_pieces = shuffle_pieces(pieces)
    return shuffled_pieces, piece_size, pieces, rows, cols

def main():
    patch_size = 8
    shuffled_pieces, piece_size, original_pieces, rows, cols = preprocessing(patch_size)
    best_solution = genetic_algorithm(shuffled_pieces, original_pieces, piece_size=piece_size, rows=rows, cols=cols)
    solved_image = reconstruct_image(best_solution, piece_size, rows, cols)
    original_image = reconstruct_image(original_pieces, piece_size, rows, cols)
    shuffled_img = reconstruct_image(shuffled_pieces, piece_size, rows, cols)

    cv2.imshow("Original Image", original_image)
    cv2.imshow("Shuffled Image", shuffled_img)
    cv2.imshow("Solved Image", solved_image)
    cv2.imwrite(f"output_images/Original_Image.png", original_image)
    cv2.imwrite(f"output_images/Shuffled_Image.png", shuffled_img)
    cv2.imwrite(f"output_images/Solved_Image.png", solved_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
