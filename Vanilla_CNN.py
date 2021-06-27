#Thanks to Victor Zhou.This code is based on his tutorial: https://victorzhou.com/blog/intro-to-cnns-par
#Thanks to Pavitrha Solai. This code is based on her tutorial also: https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
import numpy as np
import mnist


class Conv:
    def __init__(self,lr):
        self.n_filters = 8
        self.lr = lr
        self.filters = np.random.randn(self.n_filters,3,3)/self.n_filters
        self.p = 0
        self.s = 1
        self.new_w = self.new_h = int(np.floor((28 + 2*self.p - self.filters.shape[1])/self.s + 1))
        
    def iterate_regions(self,input):
        w,h = input.shape
        
        for r in range(self.new_h):
            for c in range(self.new_w):
                im_region = input[r:r+3,c:c+3]
                yield im_region,r,c

    def forward(self,input):
        self.input = input
        output = np.zeros((self.new_w,self.new_h,self.n_filters))

        for im_region,r,c in self.iterate_regions(input):

            output[r,c] = np.sum(self.filters*im_region,axis=(1,2))
        return output
    def backprop(self,dLdinput_pool):
        dLdF = np.zeros_like(self.filters)
        dLdinput_conv = np.zeros_like(self.input)
        turned_filter = self.filters.copy()
        idxs_f = [c for c in range(self.filters.shape[2])]
        idxs_turned = idxs_f.copy()
        idxs_turned.reverse()
        idxs_r = [r for r in range(self.filters.shape[1])]
        idxs_rturned = idxs_r.copy()
        idxs_rturned.reverse()
        turned_filter[:,:,idxs_f] = self.filters[:,:,idxs_turned]
        turned_filter[:,idxs_r,:] = turned_filter[:,idxs_rturned,:]
        pad_dLdinput_pool = np.pad(dLdinput_pool,((1,1),(1,1),(0,0)))
        for r in range(3):
            for c in range(3):

                dLdF[:,r,c] = np.sum(self.input[r:r+26,c:c+26][:,:,np.newaxis]*dLdinput_pool,axis=(0,1))
                dLdinput_conv = np.sum(pad_dLdinput_pool[r:r+3,c:c+3]*turned_filter.reshape(3,3,8),axis=(0,1))
        self.filters -= self.lr*dLdF
        return dLdinput_conv
        

class Pool:
    def __init__(self):
        self.p = 0
        self.s = 2
        self.fw = self.fh = 2
        self.final10 = []
    def iterative_regions(self,input):
        w,h,ls = input.shape
        new_w = new_h = int(np.floor((w + 2*self.p - self.fw)/self.s + 1))
        for r in range(new_w):
            for c in range(new_h):
                im_region = input[2*r:2*r + 2,2*c:2*c + 2]
                yield im_region,r,c
                
    def forward(self,input):
        self.input_of_pool = input
        w,h,ls = input.shape
        new_w = new_h = int(np.floor((w + 2*self.p - self.fw)/self.s + 1))
        self.input_of_pool = input
        output = np.zeros((new_h,new_w,ls))
        
        for im_region,r,c in self.iterative_regions(input):
            output[r,c] = np.max(im_region,axis=(0,1))

        return output

    def backprop(self,dLdinput_soft):
        dLdinput_of_pool = np.zeros_like(self.input_of_pool)
        for im_region,r,c in self.iterative_regions(self.input_of_pool):

            max_idxs = np.where(im_region == np.max(im_region,axis=(0,1)))
            if len(self.final10) > 2:
                self.final10.pop(0)
                self.final10.append(max_idxs)

            try:
                dLdinput_of_pool[2*r + max_idxs[0][:8],2*c + max_idxs[1][:8],max_idxs[2][:8]] = dLdinput_soft[r,c]
            except:
                print('max_idxs',max_idxs)
                
        return dLdinput_of_pool

class Softmax:
    def __init__(self,input_shape,n_classes,lr=0.001):
        
        self.lr = lr
        self.input_shape = input_shape
        
        self.n_classes = n_classes
        self.weights = np.random.randn(input_shape,self.n_classes)/input_shape

    def forward(self,input):
        self.flattened_input = input.flatten()
        self.zs = np.dot(self.flattened_input,self.weights)

        exps = np.exp(self.zs)
        self.a = exps/np.sum(exps)
        self.input_shape = input.shape 
        
        
        return self.a
    
    def calculate_loss(self,x,y):
        
        
        return -np.log(a),relevant_idx
    
    def backprop(self,x,y):
        relevant_idx =y
        y_vector = np.zeros(self.n_classes)
        y_vector[relevant_idx] = 1

        dLdz = self.a - y_vector
        dLdw = self.flattened_input[np.newaxis].T@dLdz[np.newaxis]
        dzda = self.weights
        dLda = dzda@dLdz

        self.weights -= self.lr*dLdw
        return dLda.reshape(self.input_shape)

test_images = mnist.test_images()[:1_000]
test_labels = mnist.test_labels()[:1_000]
        
conv = Conv(0.01)
pool = Pool()
soft = Softmax(13*13*8,10,lr=0.01)

def forward(image,label,lr=.005):
    out = conv.forward((image/255) - 0.5)
    out = pool.forward(out)
    out = soft.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out,loss,acc

def train(im,label,lr=.005):
    out,loss,acc = forward(im,label)
    gradient = soft.backprop(out,label)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient)
    return loss,acc

print('MNIST CNN initialized!')

corrects = []
for epoch in range(3):
    print('--- Epoch ---',epoch)
    loss = 0
    num_correct = 0
    permutation = np.random.permutation(len(test_images))
    train_images = test_images[permutation]
    train_labels = test_labels[permutation]
    for i, (im,label) in enumerate(zip(test_images,test_labels)):
        

        if i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%'%
                (i+1,loss/100,num_correct)
                )
            corrects.append(num_correct)
            loss = 0
            num_correct = 0
            
        l,acc = train(im,label)
        loss +=1
        num_correct +=acc
        

import matplotlib.pyplot as plt

plt.plot(corrects)
plt.show()
