import math
from numpy import *
import cvxpy as cp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
'''
Jigsaw Puzzle
'''


def disorImg(image,blocksize):
	# image: original image
	# blocksize: the size of the block image: blocksize x blocksize
	# blockimages: the image pieces from the original image

    [w,h] = img.size
    [colnum,rownum] = [math.floor(w/blocksize),math.floor(h/blocksize)]
    blocknum=colnum*rownum

    img2 = np.array(img)
    img2 = img2[0:rownum*blocksize,0:colnum*blocksize,:]
    
    blockimages = np.zeros((blocksize, blocksize, 3, blocknum),dtype='uint8')
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
    plt.figure(2)
    plt.imshow(disimage)
    #plt.show()
    
    return [blockimages,colnum,rownum]
                          
def computeDijo(blockimages,blocksize, n):
	# namely compute MGC
	Dijo = np.zeros((n, n, 4))
	B = 1e-6*np.array(([[0,0,1],[1,1,1],[1,0,0]]))
	B = np.tile(B,(1,1,n))
	for o in range(4):
		bitmp = np.rot90(blockimages,o-1)
		GiL = bitmp[:,blocksize-1,:,:] - bitmp[:,blocksize-2,:,:]
		GiL = np.reshape(GiL,(blocksize,3,n))
        
		uiL = np.mean(GiL,0)
		uiL = uiL.T

		invSiL = np.zeros((3,3,n))
		for i in range(n):
			SiL = np.cov(GiL[:,:,i],rowvar=False)
			if(np.linalg.det(SiL) == 0):
				print(i)
			#invSiL[:,:,i] = np.linalg.inv(SiL)

		for i in range(n):
			#tmpGil = np.reshape(GiL[:,:,i],(blocksize,3))
			#SiL = np.cov(tmpGil,rowvar=False)
			#invSiL = np.linalg.inv(SiL)
			for j in range(n):
				GijLR = bitmp[:,0,:,j] - bitmp[:,blocksize-1,:,i]
				#GijLR = np.reshape(GijLR, (blocksize,3))
				tmp = GijLR-uiL[i,:]
				multmp= np.dot(tmp,invSiL[:,:,i])
				Dijo[i,j,o] = np.trace(np.dot(multmp,tmp.T))   

def computeWijo(Dijo, n):
	Wijo = np.array((n, n ,4))
	for o in range(4):
		for i in range(n):
			for j in range(n):
				min1 = min(np.min(Dijo[0:i,j,o]),np.min(Dijo[i+1:,j,o]))
				min2 = min(np.min(Dijo[i,0:j,o]),np.min(Dijo[i,j+1:,o]))
				Wijo[i,j,o] = min(min1,min2)/Dijo[i,j,o]

	return Wijo

def computeA(Dijo, U, n):
	A = np.zeros((n,n,4))
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

	return A

def cvxSolve(Wijo,A,Delo,n,colnum,rownum):
	x = cp.Variable(n)
	hijo = cp.Variable(shape = (n,n,4))
	obj = cp.Minimize(np.sum(A*Wijo*hijo))
	tmp = np.tile(x,(1,n))
	constraints = [hijo >= (tmp - tmp.T - Delo)*A, hijo >= (tmp.T - tmp +  Delo)*A, x>=0,x<=rownum-1, y>=0, y<=colnum-1]
	prob = cp.Problem(obj, constraints)
	prob.solve()
	return x.value


def LPSolving(blocksize,blockimages,colnum,rownum):
	n = blockimages.shape[3]

	delox = [0, -1, 0, 1]
	deloy = [1, 0, -1, 0]
	DeloX = np.array((np.zeros((n,n)),-1*np.ones((n,n)),np.zeros((n,n)),np.ones((n,n))))
	DeloY = np.array((np.ones((n,n)),np.zeros((n,n)),-1*np.ones((n,n)),np.zeros((n,n))))
	Dijo = computeDijo(blockimages,blocksize, n)
	Wijo = computeWijo(Dijo, n)

	U = np.ones((n,n,4))
	A = computeA(Dijo, U, n)

	for it in range(5):
		# construct problem about x
		x = cvxSolve(Wijo,A,DeloX,n,colnum,rownum)
		y = cvxSolve(Wijo,A,DeloY,n,colnum,rownum)
		for i in range(n):
			for j in range(n):
				for o in range(4):
					if(math.abs(x[i]-x[j]-delox[o])>1e-5 or math.abs(y[i]-y[j]-deloy[o])>1e-5):
						U[i][j][o] = 0
		A = computeA(Dijo, U, n)

	return [x, y]

if __name__== '__main__':
	#read a image
	img = Image.open('2.png')
	blocksize = 28
	[blockimages,colnum,rownum] = disorImg(img,blocksize)
	[x,y] = LPSolving(blocksize,blockimages,colnum,rownum)
	print(x,y)
	#img.show()

