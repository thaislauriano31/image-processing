# SCC0251 - Image Processing and Analysis
# Assignment 04 - 2023.1
# Thaís Ribeiro Lauriano - 12542518

import imageio.v2 as imageio
import numpy as np

def flood_fill(img, seed_x, seed_y, conectivity):
    rows = len(img[0])
    cols = len(img[1])
    
    if seed_x < 0 or seed_y < 0 or seed_x >= rows or seed_y >= cols:
        return None

    start_color = img[seed_x][seed_y]
    new_color = 1 - start_color

    visited = []
    queue = []

    def valid_position(row, col):
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return False
        if img[row][col] != start_color:
            return False
        return True


    def flood_fill_util(row, col, conectivity):
        queue.append((row, col))
        visited.append((row, col))

        while queue:
            current_row, current_col = queue.pop(0)
            img[current_row][current_col] = new_color

            if conectivity == 4:
                neighbors = [(current_row-1, current_col), (current_row+1, current_col),
                            (current_row, current_col-1), (current_row, current_col+1)]
            elif conectivity == 8:
                neighbors = [(current_row-1, current_col), (current_row+1, current_col),
                            (current_row, current_col-1), (current_row, current_col+1),
                            (current_row-1, current_col-1), (current_row-1, current_col+1),
                            (current_row+1, current_col-1), (current_row+1, current_col+1)]

            for neighbor in neighbors:
                neighbor_row, neighbor_col = neighbor
                if valid_position(neighbor_row, neighbor_col) and (neighbor_row, neighbor_col) not in visited and img[neighbor_row][neighbor_col] == start_color:
                    queue.append((neighbor_row, neighbor_col))
                    visited.append((neighbor_row, neighbor_col))

    flood_fill_util(seed_x, seed_y, conectivity)
    return visited, img


def main():
    filename = input().rstrip()
    img = (imageio.imread(filename) > 127).astype(np.uint8)
    seed_x = int(input())
    seed_y = int(input())
    conectivity = int(input())

    altered_pixels, new_img = flood_fill(img, seed_x, seed_y, conectivity)

    altered_pixels = sorted(altered_pixels)
    for pixel in altered_pixels:
        print(f"({pixel[0]} {pixel[1]}) ", end='')

if __name__ == '__main__':
    main()