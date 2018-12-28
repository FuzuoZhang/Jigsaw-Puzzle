import math
#from numpy import *
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
                          
def computeDijo(blockimages,blocksize, n):
	# namely compute MGC
	Dijo = np.zeros((n, n, 4))
	B =np.zeros((3,3,n))
	for i in range(n):
		B[:,:,i] = 1e-6*np.array(([[0,0,1],[1,1,1],[1,0,0]]))
	for o in range(4):
		bitmp = np.zeros((blocksize,blocksize,3,n))
		for t in range(n):
			bitmp[:,:,:,t] = np.rot90(blockimages[:,:,:,t],o-1)
		#bitmp = np.rot90(blockimages,o-1)
		GiL = bitmp[:,blocksize-1,:,:] - bitmp[:,blocksize-2,:,:]
		#GiL = np.reshape(GiL,(blocksize,3,n))
		GiL = np.vstack(([GiL,B]))
		uiL = np.mean(GiL,0)
		uiL = uiL.T

		invSiL = np.zeros((3,3,n))
		for i in range(n):
			SiL = np.cov(GiL[:,:,i],rowvar=False)
			invSiL[:,:,i] = np.linalg.inv(SiL)

		for i in range(n):
			#tmpGil = np.reshape(GiL[:,:,i],(blocksize,3))
			#SiL = np.cov(tmpGil,rowvar=False)
			#invSiL = np.linalg.inv(SiL)
			for j in range(n):
				GijLR = bitmp[:,0,:,j] - bitmp[:,blocksize-1,:,i]
				#GijLR = np.reshape(GijLR, (blocksize,3))
				tmp = GijLR-uiL[i,:]
				#multmp= np.dot(tmp,invSiL[:,:,i])
				#Dijo[i,j,o] = np.trace(np.dot(multmp,tmp.T))  
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

'''
def cvxSolve(Wijo,A,AA,Delo,n,num):
	tmpAWijo = np.sum(A*Wijo,1)
	tmpAWijo = np.reshape(tmpAWijo,(n*4))
	i = [j for j in range(n)]
	index = np.array(([i for j in range(4)]))
	index = index.T
	print(np.shape(tmpAWijo))
	x = cp.Variable(n)
	hijo = cp.Variable(n*4)
	obj = cp.Minimize(sum(tmpAWijo*hijo))
	constraints = [hijo >= np.reshape(x[index] - x[AA] - Delo,(4*n)), hijo >= np.reshape(x[AA] - x[index] +  Delo,(4*n))]
	constraints.append(x>=0,x<=num-1)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	return x.value
'''

def cvxSolve(Wijo,A,AA,Delo,n,num):
	tmpAWijo = np.sum(np.multiply(A,Wijo),1)
	#tmpAWijo = np.reshape(tmpAWijo,(n,4))
	#np.shape(tmpAWijo)
	i = [j for j in range(n)]
	index = np.array(([i for j in range(4)]))
	index = index.T
	x = cp.Variable(n)
	hijo = cp.Variable((n,4))
	#print(np.shape(tmpAWijo),np.shape(hijo))
	obj = cp.Minimize(cp.sum(cp.multiply(tmpAWijo,hijo)))
	constraints = [hijo >= x[index] - x[AA] - Delo, hijo >= x[AA] - x[index] +  Delo,x>=0,x<=num-1]
	#constraints.append()
	#constraints[0<=x<=num-1]
	prob = cp.Problem(obj, constraints)
	prob.solve()
	return x.value


def LPSolving(blocksize,blockimages,colnum,rownum):
	n = blockimages.shape[3]

	delox = [0, -1, 0, 1]
	deloy = [1, 0, -1, 0]
	#DeloX = np.array((np.zeros((n,n)),-1*np.ones((n,n)),np.zeros((n,n)),np.ones((n,n))))
	#DeloY = np.array((np.ones((n,n)),np.zeros((n,n)),-1*np.ones((n,n)),np.zeros((n,n))))
	DeloX = np.array(([delox for i in range(n)]))
	DeloY = np.array(([deloy for i in range(n)]))
	Dijo = computeDijo(blockimages,blocksize, n)
	Wijo = computeWijo(Dijo, n)

	U = np.ones((n,n,4))
	[AA,A] = computeA(Dijo, U, n)

	for it in range(5):
		# construct problem about x
		x = cvxSolve(Wijo,A,AA,DeloX,n,rownum)
		y = cvxSolve(Wijo,A,AA,DeloY,n,colnum)
		#print(x,"\n")
		#print(y)
		for i in range(n):
			for j in range(n):
				for o in range(4):
					if(abs(x[i]-x[j]-delox[o])>1e-1 or abs(y[i]-y[j]-deloy[o])>1e-1):
						U[i][j][o] = 0
		[AA,A] = computeA(Dijo, U, n)

	return [x, y]

def LPRecover(blocksize,blockimages,colnum,rownum):
    n = blockimages.shape[3]
    [x, y] = LPSolving(blocksize,blockimages,colnum,rownum)
    recoverimg = np.zeros((rownum*blocksize,colnum*blocksize,3))
    x = np.round(x - min(x))
    y = np.round(y - min(y))
    x = x.astype(np.int)
    y = y.astype(np.int)
    for i in range(n):
        #recoverimg[y[i]*blocksize:(y[i]+1)*blocksize,x[i]*blocksize:(x[i]+1)*blocksize,:] = blockimages[:,:,:,i]
        recoverimg[x[i]*blocksize:(x[i]+1)*blocksize,y[i]*blocksize:(y[i]+1)*blocksize,:] = blockimages[:,:,:,i]
    return recoverimg

if __name__== '__main__':
	#read a image
	#img = Image.open('2.png')
	img = Image.open('8.jpg')
	#img = Image.open('lena.tif')
	blocksize = 64
	[disimage,blockimages,colnum,rownum] = disorImg(img,blocksize)
	plt.figure(1)
	plt.imshow(disimage)
	plt.show()
	#[x,y] = LPSolving(blocksize,blockimages,colnum,rownum)
	recoverimg = LPRecover(blocksize,blockimages,colnum,rownum)
	plt.figure(2)
	plt.imshow(recoverimg)
	plt.show()

