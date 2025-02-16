import math
import cvxpy as cp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
'''
Jigsaw Puzzle
'''


def disorImg(image,blocksize):
	# image: original image
	# blocksize: the size of image block: blocksize x blocksize
	# blockimages: the image pieces from the original image

    [w,h] = img.size 
    [colnum,rownum] = [math.floor(w/blocksize),math.floor(h/blocksize)]
    blocknum=colnum*rownum

    img2 = np.array(img)
    img2 = img2[0:rownum*blocksize,0:colnum*blocksize,0:3]
    
    blockimages = np.zeros((blocksize, blocksize, 3, blocknum))
    disimage = np.zeros((rownum*blocksize,colnum*blocksize,3),dtype='uint8')
    index = np.random.permutation(blocknum)
    
    for i in range(0,rownum):
    	for j in range(0,colnum):
    		xmin = i*blocksize
    		ymin = j*blocksize
    		indextmp = index[i*colnum+j]
    		blockimages[:,:,:,indextmp] = img2[xmin:xmin+blocksize, ymin:ymin+blocksize,:]
    		x=math.floor(indextmp/colnum)
    		y=indextmp - x*colnum
    		disimage[x*blocksize:(x+1)*blocksize, y*blocksize:(y+1)*blocksize,:] = blockimages[:,:,:,indextmp]
    
    return [disimage,blockimages/255,colnum,rownum]
                          
def computeDijo(blockimages, blocksize, n):
	# namely compute MGC
	Dijo = np.zeros((n, n, 4))
	B =np.zeros((3,3,n))
	for i in range(n):
		B[:,:,i] = 1e-6*np.array(([[0,0,1],[1,1,1],[1,0,0]]))
	for o in range(4):
		bitmp = np.zeros((blocksize,blocksize,3,n))
		for t in range(n):
			bitmp[:,:,:,t] = np.rot90(blockimages[:,:,:,t],o-1)
			B[:,:,t] = np.rot90(B[:,:,t],o-1)
		GiL = bitmp[:,blocksize-1,:,:] - bitmp[:,blocksize-2,:,:]
		GiL = np.vstack(([GiL,B]))
		uiL = np.mean(GiL,0)
		uiL = uiL.T

		invSiL = np.zeros((3,3,n))
		for i in range(n):
			SiL = np.cov(GiL[:,:,i],rowvar=False)
			invSiL[:,:,i] = np.linalg.inv(SiL)

		for i in range(n):
			for j in range(n):
				GijLR = bitmp[:,0,:,j] - bitmp[:,blocksize-1,:,i]
				tmp = GijLR-uiL[i,:]
				Dijo[i,j,o] = np.trace(tmp@invSiL[:,:,i]@tmp.T) 
	return Dijo

def computeWijo(Dijo, n):
	Wijo = np.zeros((n, n ,4))
	for o in range(4):
		for i in range(n):
			for j in range(n):
				if (i > 0 and i < n-1):
					min1 = min(np.min(Dijo[0:i,j,o]),np.min(Dijo[i+1:,j,o]))
				if (i <= 0):
					min1 = np.min(Dijo[1:,j,o])
				if (i >= n-1):
					min1 = np.min(Dijo[:n-1,j,o])
				if (j > 0 and j < n-1):
					min2 = min(np.min(Dijo[i,0:j,o]),np.min(Dijo[i,j+1:,o]))
				if (j <= 0):
					min2 = np.min(Dijo[i,1:,o])
				if (j >= n-1):
					min2 = np.min(Dijo[i,:n-1,o])
				Wijo[i,j,o] = min(min1,min2)/Dijo[i,j,o]

	return Wijo

def computeA(Dijo, U, n):
	A = np.zeros((n,n,4))
	AA = np.zeros((n,4),dtype=int)
	for i in range(n):
		for o in range(4):
			j = -1
			minDijo = float("inf")
			for k in range(n):
				if(U[i,k,o]>0 and Dijo[i,k,o]<minDijo):
					j = k
					minDijo = Dijo[i,k,o]
			if(j>-1):
				A[i,j,o] = 1
				AA[i,o] = j

	return [AA,A]

def cvxSolve(Wijo,A,AA,Delo,n,num):
	tmpAWijo = np.sum(np.multiply(A,Wijo),1)
	i = [j for j in range(n)]
	index = np.array(([i for j in range(4)]))
	index = index.T
	x = cp.Variable(n)
	hijo = cp.Variable((n,4))
	obj = cp.Minimize(cp.sum(cp.multiply(tmpAWijo,hijo)))
	constraints = [hijo >= x[index] - x[AA] - Delo, hijo >= x[AA] - x[index] +  Delo,x>=0,x<=num-1]
	prob = cp.Problem(obj, constraints)
	prob.solve()
	return x.value

def LPSolving(blocksize,blockimages,colnum,rownum):
	n = blockimages.shape[3]

	deloy = [0, -1, 0, 1]
	delox = [1, 0, -1, 0]
	DeloX = np.array(([delox for i in range(n)]))
	DeloY = np.array(([deloy for i in range(n)]))
	Dijo = computeDijo(blockimages,blocksize, n)
	Wijo = computeWijo(Dijo, n)

	U = np.ones((n,n,4))
	[AA,A] = computeA(Dijo, U, n)

	for it in range(10):
		x = cvxSolve(Wijo,A,AA,DeloX,n,rownum)
		y = cvxSolve(Wijo,A,AA,DeloY,n,colnum)
		for i in range(n):
			for j in range(n):
				for o in range(4):
					if(abs(x[i]-x[j]-delox[o])>5e-1 or abs(y[i]-y[j]-deloy[o])>5e-1):
						U[i][j][o] = 0
		[AA,A] = computeA(Dijo, U, n)

	return [x, y]

def LPRecover(blocksize,blockimages,colnum,rownum):
    n = blockimages.shape[3]
    [x, y] = LPSolving(blocksize,blockimages,colnum,rownum)
    recoverimg = np.ones((rownum*blocksize,colnum*blocksize,3))
    x = np.round(x - min(x))
    y = np.round(y - min(y))
    x = x.astype(np.int)
    y = y.astype(np.int)
    for i in range(n):
        recoverimg[x[i]*blocksize:(x[i]+1)*blocksize,y[i]*blocksize:(y[i]+1)*blocksize,:] = blockimages[:,:,:,i]
    return recoverimg

if __name__== '__main__':
	#read a image
	img = Image.open('./images/lena.tif')
	blocksize = 64
	[disimage,blockimages,colnum,rownum] = disorImg(img,blocksize)
	recoverimg = LPRecover(blocksize,blockimages,colnum,rownum)
	plt.figure(1)
	plt.imshow(disimage)
	plt.show()
	plt.figure(2)
	plt.imshow(recoverimg)
	plt.show()

