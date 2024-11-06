import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# Okuma Fonksiyonu
def read_image():
    img_gray = cv2.imread("/home/emre/Fall/MachineLearning/homework1/test.jpg", cv2.IMREAD_GRAYSCALE)
    return img_gray

# Resmi 8 Parçaya Ayırma Fonksiyonu
def slice_eight(img):
    rows = 4
    cols = 2
    height, width = img.shape

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

    return pieces, (piece_height, piece_width)

# Parçaları Karıştırma Fonksiyonu
def shuffle_pieces(pieces):
    shuffled_pieces = pieces.copy()
    random.shuffle(shuffled_pieces)
    return shuffled_pieces

# Resmi Yeniden Yapılandırma Fonksiyonu
def reconstruct_image(pieces, piece_size):
    rows, cols = 4, 2
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

# Fitness Fonksiyonu
def fitness_function(individual, original_pieces):
    correct_positions = sum(1 for i in range(len(individual)) if np.array_equal(individual[i], original_pieces[i]))
    return correct_positions

# Ebeveyn Seçme Fonksiyonu
def select_parents(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[-2:]  # En iyi iki bireyi seç
    return [population[selected_indices[0]], population[selected_indices[1]]]

# Çaprazlama Fonksiyonu
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    
    def is_piece_in_list(piece, pieces_list):
        return any(np.array_equal(piece, p) for p in pieces_list)
    
    child = parent1[:cut] + [piece for piece in parent2 if not is_piece_in_list(piece, parent1[:cut])]
    return child

# Mutasyon Fonksiyonu
def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Fitness Skorları Grafiği
def plot_fitness_scores(generation, fitness_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(fitness_scores)), fitness_scores, color='blue')
    plt.xlabel('Birey')
    plt.ylabel('Fitness Skoru')
    plt.ylim(0, 8)  # Maksimum fitness skoru 8 olacak şekilde ayarla
    plt.title(f'Generation {generation} Fitness Scores')
    
    # X ekseninde tam sayı etiketler göster
    plt.xticks(range(len(fitness_scores)), range(1, len(fitness_scores) + 1))
    
    plt.savefig(f"output_graph/Generation_{generation}_fitness_scores.png")
    plt.close()


# Genetik Algoritma

def genetic_algorithm(shuffled_pieces, original_pieces, piece_size, population_size=20, generations=1000):
    population = [shuffle_pieces(shuffled_pieces) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [fitness_function(individual, original_pieces) for individual in population]
        
        # Fitness skorları grafiğini kaydet
        plot_fitness_scores(generation, fitness_scores)
        
        # Geçerli nesil için fitness skorlarını yazdır
        print(f"Generation {generation}: Fitness Scores = {fitness_scores}")
        
        if max(fitness_scores) == len(original_pieces):  # Tam doğru birey bulunduysa
            best_individual = population[np.argmax(fitness_scores)]
            print(f"Solved in generation {generation}")
            return best_individual

        parents = select_parents(population, fitness_scores)
        population_best_individual = reconstruct_image(parents[0], piece_size)
        cv2.imshow(f"Generation {generation} best individual", population_best_individual)
        cv2.imwrite(f"output_images/Generation_{generation}_best_individual.png", population_best_individual)
        next_generation = []

        for _ in range(population_size):
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            next_generation.append(child)

        population = next_generation

    best_individual = population[np.argmax([fitness_function(ind, original_pieces) for ind in population])]
    return best_individual

# Önişleme Fonksiyonu
def preprocessing():
    img = read_image()
    pieces, piece_size = slice_eight(img)
    shuffled_pieces = shuffle_pieces(pieces)
    return shuffled_pieces, piece_size, pieces

# Ana Fonksiyon
def main():
    shuffled_piece, piece_size, original_pieces = preprocessing()
    best_solution = genetic_algorithm(shuffled_piece, original_pieces, piece_size=piece_size)
    solved_image = reconstruct_image(best_solution, piece_size)
    original_image = reconstruct_image(original_pieces, piece_size)
    shuffled_img = reconstruct_image(shuffled_piece, piece_size)

    cv2.imshow("Original Image", original_image)
    cv2.imshow("Shuffled Image", shuffled_img)
    cv2.imshow("Solved Image", solved_image)
    cv2.imwrite(f"output_images/Shuffled_Image.png", shuffled_img)
    cv2.imwrite(f"output_images/Solved_Image.png", solved_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Başlatılırsa Ana Fonksiyonu Çağır
if __name__ == "__main__":
    main()
