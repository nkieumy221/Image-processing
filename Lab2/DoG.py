import cv2
import numpy as np
import matplotlib.pyplot as plt
# Reference: Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions by Xiaoyang Tan and Bill Triggs
# https://lear.inrialpes.fr/pubs/2007/TT07/Tan-amfg07a.pdf

# read image as grayscale float in range 0 to 1
img = cv2.imread('Lab2/anh.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

# set arguments
gamma = 0.2
alpha = 0.1
tau = 3.0

# gamma correction
img_gamma = np.power(img, gamma)
img_gamma2 = (255.0 * img_gamma).clip(0,255).astype(np.uint8)

# DOG
blur1 = cv2.GaussianBlur(img_gamma, (0,0), 1, borderType=cv2.BORDER_REPLICATE)
blur2 = cv2.GaussianBlur(img_gamma, (0,0), 2, borderType=cv2.BORDER_REPLICATE)
img_dog = (blur1 - blur2)
# normalize by the largest absolute value so range is -1 to 
img_dog = img_dog / np.amax(np.abs(img_dog))
img_dog2 = (255.0 * (0.5*img_dog + 0.5)).clip(0,255).astype(np.uint8)

# contrast equalization equation 1
img_contrast1 = np.abs(img_dog)
img_contrast1 = np.power(img_contrast1, alpha)
img_contrast1 = np.mean(img_contrast1)
img_contrast1 = np.power(img_contrast1,1.0/alpha)
img_contrast1 = img_dog/img_contrast1

# contrast equalization equation 2
img_contrast2 = np.abs(img_contrast1)
img_contrast2 = img_contrast2.clip(0,tau)
img_contrast2 = np.mean(img_contrast2)
img_contrast2 = np.power(img_contrast2,1.0/alpha)
img_contrast2 = img_contrast1/img_contrast2
img_contrast = tau * np.tanh((img_contrast2/tau))

# Scale results two ways back to uint8 in the range 0 to 255
img_contrastA = (255.0 * (img_contrast+0.5)).clip(0,255).astype(np.uint8)
img_contrastB = (255.0 * (0.5*img_contrast+0.5)).clip(0,255).astype(np.uint8)

# show results
#Tạo vùng vẽ tỷ lệ 16:9
fig = plt.figure(figsize=(16, 9))
#Tạo 9 vùng vẽ con, phân bố 3 hàng 3 cột
(ax1, ax2, ax3), (ax4,ax5, ax6) = fig.subplots(2, 3)
ax1.imshow(img, cmap='gray')
ax1.set_title("ảnh gốc")

ax2.imshow(img_gamma2, cmap='gray')
ax2.set_title("Gamma")

ax3.imshow(img_dog2, cmap='gray')
ax3.set_title("DoG")

ax4.imshow(img_contrast1, cmap='gray')
ax4.set_title("CE1")

ax5.imshow(img_contrastA, cmap='gray')
ax5.set_title("CE_A")

ax6.imshow(img_contrastB, cmap='gray')
ax6.set_title("CE_B")

plt.show()
""" cv2.imshow('Face', img)
cv2.imshow('Gamma', img_gamma2)
cv2.imshow('DoG', img_dog2)
cv2.imshow('CE1', img_contrast1)
cv2.imshow('CE_A', img_contrastA)
cv2.imshow('CE_B', img_contrastB)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save results
cv2.imwrite('face_contrast_equalization_A.jpg', img_contrastA)
cv2.imwrite('face_contrast_equalization_B.jpg', img_contrastB) """