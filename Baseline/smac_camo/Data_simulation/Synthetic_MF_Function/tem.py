import numpy as np


class Non_Linear_Sin():
    def __init__(self, debug=False):
        self.dim = 1
        self.flevels = 2
        self.maximum = 0.0
        

        self.bounds = ((0,1.5))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
            
        self.Flist = []
        self.Flist.append(self.eval_fed_L1)
        self.Flist.append(self.eval_fed_L2)

        
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)

        # return -ym
        return ym
    
    def eval_fed_L1(self, xn):
        
        x1 = xn[0]
        # x2 = xn[1]
        # torch.sin(xtr * 8 * math.pi)
        f = np.sin(x1 * 8 * np.pi)

        
        return f
    
    def eval_fed_L2(self, xn):
        x1 = xn[0]
        
        f3 = self.eval_fed_L1(xn)
        
        f2 = (x1 - np.sqrt(2)) * (f3 ** 2)

        return f2
    
class Forrester():
    def __init__(self, debug=False):
        self.dim = 1
        self.flevels = 2
        self.maximum = 50.0
        

        self.bounds = ((0,1.5))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
            
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)

        
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)

        # return -ym
        return ym
    
    
    def eval_fed_L1(self, xn):
        x1 = xn[0]

        f = (6 * x1) ** 2 * np.sin(12 * x1 - 4)
        
        return f
    
    def eval_fed_L0(self, xn):
        x1 = xn[0]
        
        f = 0.5 * self.eval_fed_L1(xn) + 10 * (x1 - 0.5) + 5

        return f
    

if __name__ == "__main__":
    model = Forrester()

    X = [np.random.rand(10)]
    print(model.eval_fed_L0(X))
