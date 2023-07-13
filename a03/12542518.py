# SCC0251 - Image Processing and Analysis
# Assignment 03 - 2023.1
# Thaís Ribeiro Lauriano - 12542518

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import convolve

# Recebe oos caminhos das imagens a serem carregadas e retorna um vetor com as imagens carregadas
def get_imgs(img_paths):
    img_vector = []
    for img_path in img_paths:
        img_vector.append(imageio.imread(img_path))
    
    return img_vector

# Converte imagens coloridas para escala de cinza
def color_to_bnw(imgs):
    imgs_gray = []
    for img in imgs:
        imgs_gray.append(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]))

    return imgs_gray

# Calcula os gradientes horizontal e vertical,
# em seguida computa as matrizes de magnitude e oritntação e as retorna 
def compute_gradient(image):
    # Calculando gradiente horizontal
    grad_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x_image = convolve(image, grad_x)
    
    # Calculando gradiente vertical
    grad_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    grad_y_image = convolve(image, grad_y)
    
    # Calculando as matrizes de magnitude do gradiente e orientação (phi)
    magnitude = np.sqrt(grad_x_image**2 + grad_y_image**2)/np.sqrt(grad_x_image**2 + grad_y_image**2).sum()
    orientation = np.degrees(np.arctan2(grad_y_image, grad_x_image)) % 180
    
    return magnitude, orientation

# Computa a matriz de bins a partir da matriz de orientação
def compute_bins(orientation):
    bins = np.mod((orientation+10), 180) // 20
    return bins

# Computa o desctritor para o hog a partir da magnitude e da matriz de bins
def compute_descriptor(bins, magnitude):
    descriptor, _ = np.histogram(bins, bins=9, range=(0, 9), weights=magnitude)
    return descriptor

# Computa o hog para uma imagem
def hog(image):
    magnitude, orientation = compute_gradient(image)
    hog = compute_descriptor(compute_bins(orientation), magnitude)
   
    return hog

# Classificação k-NN
def knn(X_train, y_train, X_test, k):
    distances = np.sqrt(np.sum((X_test - X_train)**2, axis=1))
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    counts = np.bincount(nearest_labels)
    return np.argmax(counts)

def main():
    # Lendo os nomes dos arquivos, carregando as imagens e convertendo-as para preto&branco
    non_human_paths = input().rstrip().split(sep=' ')
    nonhuman_rgb = get_imgs(non_human_paths)
    non_human_imgs = color_to_bnw(nonhuman_rgb)
    
    human_paths = input().rstrip().split(sep=' ')
    human_rgb = get_imgs(human_paths)
    human_imgs = color_to_bnw(human_rgb)

    test_paths = input().rstrip().split(sep=' ')
    test_imgs = get_imgs(test_paths)
    test_imgs = color_to_bnw(test_imgs)

    # Treinando o modelo
    X = []
    y = []
    for image in human_imgs:
        hog_descriptor = hog(image)
        X.append(hog_descriptor)
        y.append(1) # classe positiva (humano)

    for image in non_human_imgs:
        hog_descriptor = hog(image)
        X.append(hog_descriptor)
        y.append(0) # classe negativa (não humano)

    X = np.array(X)
    y = np.array(y)

    # Teste de classificação
    predictions = []
    for test_image in test_imgs:
        test_descriptor = hog(test_image)
        prediction = knn(X, y, test_descriptor, k=3)
        predictions.append(prediction)

    print(*predictions)

if __name__ == '__main__':
    main()