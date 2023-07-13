# SCC0251 - Image Processing and Analysis
# Assignment 01 - 2023.1
# Thaís Ribeiro Lauriano - 12542518

import imageio.v2 as imageio
import numpy as np

# checks if two images are the same shape/resolution
def same_shape(img1, img2):
    if img1.shape != img2.shape:
        print("Images aren't the same resolution")
        return False
    return True

def superresolution(imgs):
    n = imgs[0].shape
    if not same_shape(imgs[0], imgs[1]) or not same_shape(imgs[0], imgs[2]) or not same_shape(imgs[0], imgs[3]):
        return

    gen_img = np.zeros((2*n[0], 2*n[1]))

    for i in range(n[0]):
        for j in range(n[1]):
            gen_img[i*2][j*2] = imgs[0][i][j]
            gen_img[i*2+1][j*2] = imgs[1][i][j]
            gen_img[i*2][j*2+1] = imgs[2][i][j]
            gen_img[i*2+1][j*2+1] = imgs[3][i][j]
    
    return gen_img

# builds the histogram for a single image
def histogramSI(img):
    hist = np.zeros(256).astype(int)
    for i in range(256):
        hist[i] = np.sum(img == i)
            
    return hist

# builds the histogram for the 4 low resolution images together
def histogramJ(imgs):
    hist = np.zeros(256).astype(int)
    
    for i in range(4):
        for j in range(256):
            hist[j] = np.sum(imgs[i] == j)

    return hist

# calculates the cumulative histogram for any given histogram
def cumulative_histogram(hist):   
    histC = np.zeros(256).astype(int)

    histC[0] = hist[0] 
    for i in range(1,  256):
        histC[i] = hist[i] + histC[i-1]

    return histC

def histogram_equalization(img, histC):
    n = img.shape
    
    img_eq = np.zeros([n[0],n[1]]).astype(int)
     
    for input in range(256):
        output = ((256-1)/float(n[0]*n[1]))*histC[input]
        
        img_eq[ np.where(img == input) ] = output
    
    return img_eq

def single_img_cumul_hist(imgs):
    eq_imgs = []
    for i in range(4): 
        hist = histogramSI(imgs[i])
        histC = cumulative_histogram(hist)
        eq_imgs.append(histogram_equalization(imgs[i],histC))

    return eq_imgs

def joint_cumulative_hist(imgs):
    eq_imgs = []
    
    hist = histogramJ(imgs)
    histC = cumulative_histogram(hist)
    for i in range(4):
        eq_imgs.append(histogram_equalization(imgs[i], histC))
    return eq_imgs

def gamma_correction(imgs, gamma):
    n = imgs[0].shape
    if not same_shape(imgs[0], imgs[1]) or not same_shape(imgs[0], imgs[2]) or not same_shape(imgs[0], imgs[3]):
        return

    new_imgs = np.zeros([4,n[0],n[1]]).astype(int)
    for k in range(4):
        for i in range(n[0]):
            for j in range(n[1]):
                new_imgs[k][i][j] = np.floor(255 * (imgs[k][i][j]/255.0)**(1/gamma))

    return new_imgs
                
 # RMSE
def comparison(gen_img, ref_img):
    n = gen_img.shape
    if not same_shape(gen_img, ref_img):
        return
    sum = 0
    for i in range(n[0]):
        for j in range(n[1]):
            sum += (ref_img[i][j] - gen_img[i][j])**2
            
    error = np.sqrt(sum/(n[0]*n[1]))
    print("{:.4f}".format(error))
    return

def main():
    # reading image names and loading image files
    imglow_bn = input()

    img_low = []
    for i in range(4):
        img_low.append(imageio.imread(imglow_bn+str(i)+'.png'))

    imghigh_name = input()
    img_high = imageio.imread(imghigh_name)

    # reading F (enhancement type) and gamma
    F = int(input())
    gamma = float(input())

    # applying chosen enhancement method (result is stored in 'new_imgs')
    if F > 3 or F < 0:
        print("Invalid value of F")
    elif F == 0:
        # no enhancement method applied
        new_imgs = img_low
    elif F == 1:
        new_imgs = single_img_cumul_hist(img_low)
    elif F == 2:
        new_imgs = joint_cumulative_hist(img_low)
    elif F == 3:
        new_imgs = gamma_correction(img_low, gamma)

    comparison(superresolution(new_imgs), img_high)

if __name__ == '__main__':
    main()