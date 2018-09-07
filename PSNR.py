import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math

AvgPSNR = 0  # initialise average PSNR Value
givenimg = ["cameraman","house","jetplane","lake","lena_gray_512","livingroom","mandril_gray","peppers_gray","pirate","walkbridge","woman_blonde","woman_darkhair"]
PNSR = []
for start in range (0,12):
    
    img = cv2.imread(f"standard_test_images/{givenimg[start]}.tif", 0) #since the image is grayscale, we need only one channel and the value '0' indicates just that
    img = np.array(img)/255
    
    SmallImg = [0] * 256  # making a small image of size 256 X 256
    for i in range(256):
        SmallImg[i] = [0] * 256
    


    FinalImg = [0] * 512 # making final imagr of size 512 X 512
    for i in range(512):
        FinalImg[i] = [0] * 512
    for i in range (0,512,2):               # initialising alternate rows and colums 
        for j in range (0,512,2):
            FinalImg[i][j]=img[i][j]
            SmallImg[i//2][j//2] = img[i][j]
    SmallImg = np.array(SmallImg)
# 

    windowsize=16      # making a window of size 16 * 16 pixels

    for i in range (0,256,windowsize):
        for j in range (0,256,windowsize):
            window = [0]*windowsize
            for k in range (windowsize):
                window[k]=[0]*windowsize
            for p in range (i,i+windowsize):
                for q in range (j,j+windowsize):
                    window[p%windowsize][q%windowsize]=SmallImg[p][q]

            X=[]     # X matrice for estimating alpha 1
            X2=[]    # X2 matrice for estimating alpha 2
            Y=[]     # Y matrice 
            for p in range (0,windowsize-2):          # filling X, X2 and Y matrices
                for q in range (0,windowsize-2):
                    Y.append([window[p+1][q+1]])
                    X.append([window[p][q],window[p][q+2],window[p+2][q],window[p+2][q+2]])
                    X2.append([window[p][q+1],window[p+1][q+2],window[p+2][q+1],window[p+1][q]])
            
            T_X = np.transpose(X)    # transpose of X 
            T_X2 = np.transpose(X2)  # transpose of X2
            
            alpha = np.array(np.matmul(np.linalg.pinv(np.matrix(np.matmul(T_X, X))),np.matmul(T_X,Y))) # estimating parameters for diagnols using (X^T X)^-1 (X^T Y)
           
            alpha2 = np.array(np.matmul(np.linalg.pinv(np.matrix(np.matmul(T_X2, X2))),np.matmul(T_X2,Y))) # estimating parameters for UP DOWNS LEFT RIGHT using (X^T X)^-1 (X^T Y)


            for p in range(2*i +1,2*i + 2*windowsize,2):  # Filling image using window of small image (Filling only diagols)
                for q in range (2*j +1,2*j + 2*windowsize,2):
                    if p<511 and q<511:
                        FinalImg[p][q] = alpha[0][0]*FinalImg[p-1][q-1] + alpha[1][0]*FinalImg[p-1][q+1] + alpha[2][0]*FinalImg[p+1][q-1] + alpha[3][0]*FinalImg[p+1][q+1]
                    elif p>=511 and q<511:
                        FinalImg[p][q] = alpha[0][0]*FinalImg[p-1][q-1] + alpha[1][0]*FinalImg[p-1][q+1] 
                    elif p<511 and q>=511:
                        FinalImg[p][q] = alpha[0][0]*FinalImg[p-1][q-1] + alpha[2][0]*FinalImg[p+1][q-1]

            for p in range(2*i+1,2*i + 2*windowsize):   # Filling image using window of small image (Filling Up downs)
                for q in range (2*j +1,2*j + 2*windowsize):
                    if (p+q)%2 != 0 and p<511 and q<511:
                        FinalImg[p][q]=alpha2[0][0]*FinalImg[p-1][q] + alpha2[1][0]*FinalImg[p][q+1] + alpha2[2][0]*FinalImg[p+1][q] + alpha2[3][0]*FinalImg[p][q-1]
                    elif (p+q)%2 != 0 and p<511 and q == 511:
                        FinalImg[p][q]=alpha2[0][0]*FinalImg[p-1][q] + alpha2[2][0]*FinalImg[p+1][q] + alpha2[3][0]*FinalImg[p][q-1]
                    elif (p+q)%2 != 0 and p == 511 and q<511:
                        FinalImg[p][q]=alpha2[0][0]*FinalImg[p-1][q] + alpha2[1][0]*FinalImg[p][q+1] + alpha2[3][0]*FinalImg[p][q-1]
                    
            for q in range(2*j +1,2*j + 2*windowsize,2):  # Filling corner cases
                if i>0  and q<511:
                    FinalImg[2*i][q]=alpha2[0][0]*FinalImg[2*i -1][q] + alpha2[1][0]*FinalImg[2*i][q+1] + alpha2[2][0]*FinalImg[2*i +1][q] + alpha2[3][0]*FinalImg[2*i][q-1]
                elif i==0 and q<511:
                    FinalImg[2*i][q]=alpha2[1][0]*FinalImg[2*i][q+1] + alpha2[2][0]*FinalImg[2*i +1][q] + alpha2[3][0]*FinalImg[2*i][q-1]
                elif i==0 and q==511:
                    FinalImg[2*i][q]=alpha2[2][0]*FinalImg[2*i +1][q] + alpha2[3][0]*FinalImg[2*i][q-1]
                elif i>0 and q==511:
                    FinalImg[2*i][q]=alpha2[0][0]*FinalImg[2*i -1][q] + alpha2[2][0]*FinalImg[2*i +1][q] + alpha2[3][0]*FinalImg[2*i][q-1]
            for p in range(2*i +1,2*i + 2*windowsize,2):
                if j>0  and p<511:
                    FinalImg[p][2*j]=alpha2[0][0]*FinalImg[p-1][2*j] + alpha2[1][0]*FinalImg[p][2*j+1] + alpha2[2][0]*FinalImg[p +1][2*j] + alpha2[3][0]*FinalImg[p][2*j-1]
                elif j==0 and p<511:
                    FinalImg[p][2*j]=alpha2[0][0]*FinalImg[p-1][2*j] + alpha2[1][0]*FinalImg[p][2*j+1] + alpha2[2][0]*FinalImg[p +1][2*j]
                elif j==0 and p==511:
                    FinalImg[p][2*j]=alpha2[0][0]*FinalImg[p-1][2*j] + alpha2[1][0]*FinalImg[p][2*j+1] 
                elif j>0 and p==511:
                    FinalImg[p][2*j]=alpha2[0][0]*FinalImg[p-1][2*j] + alpha2[1][0]*FinalImg[p][2*j+1] + alpha2[3][0]*FinalImg[p][2*j-1]

    for j in range(512):              #cropping the image 
        FinalImg[511][j]=img[511][j]
        FinalImg[0][j] = img[0][j]

    for i in range(512):
        FinalImg[i][511]=img[i][511]
        FinalImg[i][0]=img[i][0]


    MSE = 0                         #calculating PSNR
    for i in range (0,512):
        for j in range (0,512):
            MSE = MSE + 255*255*((img[i][j] - FinalImg[i][j])*(img[i][j] - FinalImg[i][j]))
    MSE = ((MSE)/(512*512))
    # print(MSE)
    PNSRlocal = 10*math.log10(((255*255)/(MSE)))
    PNSR.append(PNSRlocal)
    print(givenimg[start],end=' ') #print PSNR
    print(PNSRlocal)

for i in range (0,12):            #calculating average PSNR
    AvgPSNR = AvgPSNR +  PNSR[i]
AvgPSNR = AvgPSNR/12    
print("Average PSNR ",end=' ')
print(AvgPSNR)                   #printing average PSNR

                                 #plot the PSNR Graph
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],[PNSR[0], PNSR[1], PNSR[2], PNSR[3],PNSR[4],PNSR[5],PNSR[6],PNSR[7],PNSR[8],PNSR[9],PNSR[10]])

