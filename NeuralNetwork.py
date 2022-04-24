import numpy as np
import numpy as np
from tabulate import tabulate
from numpy import transpose as trans

def Sig(x):
    return 1.0/(1 + np.exp(-x))
def dSig(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y, h1, l):
        self.x  = x #In
        self.y  = y #Out
        #Number of Neurons
        self.h1 = h1
        #Learning Rate
        self.l  = l
        #Activations
        self.a2 = np.zeros(self.y.shape)
        self.a1 = np.zeros((self.h1, self.x.shape[1]))
        #Biases
        self.b1 = np.random.rand(self.h1, 1)
        self.b2 = np.random.rand(self.y.shape[0],1)
        #Weights
        self.w1 = np.random.rand(self.h1, self.y.shape[0])
        self.w2 = np.random.rand(self.y.shape[0], self.h1)

        ## For AND Gate ##
        # print(self.a1.shape)#A1 (h1,3)
        # print(self.a2.shape)#A2 (8,1)
        # print(self.y.shape) #Y  (8,1)
        # print(self.x.shape) #X  (8,3)
        # print(self.w1.shape)#w1 (h1,8)
        # print(self.w2.shape)#w2 (8,h1)

    def FeedForward(self):
        self.a1 = Sig(np.dot(self.w1, self.x) + self.b1)
        self.a2 = Sig(np.dot(self.w2, self.a1) + self.b2)

    def BackProp(self):
        dw2 = np.dot(self.a1, ((self.y - self.a2) * dSig(self.a2)).T) * self.l
        dw1 = np.dot(self.x, (np.dot(trans((self.y - self.a2) * dSig(self.a2)), self.w2)) * dSig(self.a1.T)) * self.l

        self.w2 += dw2.T
        self.w1 += dw1.T

        #db2 = np.dot(self.w2.T, (2*(self.y - self.a2) * dSig(self.a2)))
        #self.b2 += db2
        # self.b1 += db1

    def Run(self, nn, output, epochs):

        for i in range(epochs):
            nn.FeedForward()
            nn.BackProp()

            if (i == epochs-1):
                table = [[y[0],a2[0]] for y, a2 in zip(output, nn.a2)]

                titles = ["Correct Values", "AI Predicted Values"]
                print("\n")
                print("            After {} epochs".format(epochs))
                print(tabulate(table, titles, tablefmt="github"))
                print("\n")
                # print(output)



andIn = np.array([  [0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1] ])
andOut = np.array([ [0],
                    [0],
                    [0],
                    [1],
                    [1],
                    [.7],
                    [0],
                    [.2]])

# nn = NeuralNetwork(andIn, andOut, 20, .5)
# nn.Run(nn, andOut,20000)