from dlroms import*
from torch.nn import Parameter
from torch import sigmoid, logsumexp, log
import numpy as np

class PushForward(DFNN):
    def __init__(self, *args, **kwargs):
        super(PushForward, self).__init__(*args, **kwargs)
        self.h = Parameter(self.coretype().tensor(0.1 if 'h0' not in kwargs.keys() else kwargs['h0']))
        self.htrainable = True
        self.xscale = lambda x: x
        self.yscale = lambda y: y
        if("m" not in kwargs.keys()):
            self.m = None
        self.extras = dict()

    def forward(self, x):
        phi = self[0]
        psi = self[1]
        Y = phi(self.xscale(x))
        B, r, q = Y.shape
        u = self.coretype().randn(B, self.m, q) 
        U = psi(u.reshape(-1, q)).reshape(B, self.m, r, q)
        output = (Y.unsqueeze(1)*U).sum(axis = -2)
        return self.yscale(output)

    def sample(self, x, m):
        try:
            self.m = m
            ysampled = self.forward(x)
            return ysampled + self.get_h()*self.coretype().tensor(np.random.randn(*ysampled.shape))
        except TypeError as e:
            return self.sample(self.coretype().tensor(x), m).detach().cpu().numpy()
    
    def parameters(self):
        return super(PushForward, self).parameters() + ([self.h] if self.htrainable else [])

    def get_h(self):
        return self.h

    def set_h(self, h):
        self.h = Parameter(self.coretype().tensor(h))

    def callback(self):
        h = self.get_h().detach().cpu().numpy()
        return "\nCurrent h: [%s]" % (("%.2e, "*len(h)) % tuple(h))[:-2]
    
    def density(self, ysample, yval):
        h = self.get_h()
        c = (2*np.pi)**(0.5*yval.shape[-1])
        d = ysample - yval.unsqueeze(1)
        e = (-0.5*(d/h).pow(2).sum(axis = -1)).exp()
        return e.mean(axis = -1)/((h.prod())*c)

    def neglogdensity(self, ysample, yval):
        h = self.get_h()
        c = (2*np.pi)**(0.5*yval.shape[-1])
        d = (ysample - yval.unsqueeze(1))
        a = (0.5*(d/h)**2).sum(axis = -1)
        n = a.shape[-1]
        return -logsumexp(-a, axis = -1) + log((h.prod())*c*n)

    def h_freeze(self):
        self.h.requires_grad_(False)
        self.htrainable = False

    def h_unfreeze(self):
        self.h.requires_grad_(True)
        self.htrainable = True

    def wb_freeze(self):
        super(PushForward, self).freeze()
    
    def wb_unfreeze(self):
        super(PushForward, self).unfreeze()
    
    def freeze(self):
        self.wb_freeze()
        self.h_freeze()
    
    def unfreeze(self):
        self.wb_unfreeze()
        self.h_unfreeze()
    
    def dictionary(self, label = ""):
        params = super(PushForward, self).dictionary(label)
        params.update({'h':self.h.detach().cpu().numpy()})
        params.update(self.extras)
        return params

    def save(self, path, label = ""):
        np.savez(path.replace(".npz", ""), **self.dictionary(label))

    def load(self, path, label = ""):
        loaded = np.load(path.replace(".npz", "") + ".npz")
        self[0].load(path, label = "1")
        self[1].load(path, label = "2")
        self.set_h(loaded['h'])
        for key in loaded.keys():
            if("ex_" in key):
                self.extras[key] = loaded[key]

        xmean, xsd = self.coretype().tensor(self.extras['ex_xmean'], self.extras['ex_xsd'])
        ymean, ysd = self.coretype().tensor(self.extras['ex_ymean'], self.extras['ex_ysd'])
        self.yscale = lambda u: u*ysd + ymean
        self.xscale = lambda xv: (xv-xmean)/xsd

    def write(self, label = ""):
        params = super(PushForward, self).write(label)
        params.update({'h':self.h.detach().cpu().numpy()})
        return params
    
    def read(self, params, label = ""):
        super(PushForward, self).read(params, label)
        self.h = Parameter(self.coretype().tensor(params['h']))

    def fit(self, x, y, initialize = True, h0 = 0.1, delta = 1e-15, m = None, **train_args):
        if(initialize):
            self.He()
            
        self.set_h([h0]*y.shape[-1])
        if("optim" not in train_args.keys()):
            from torch.optim import Adam
            train_args["optim"] = Adam
        if("keepbest" not in train_args.keys()):
            train_args["keepbest"] = True
        if(not(m is None)):
            self.m = m
        
        def loss(ytrue, ypred):
              return -(delta + self.density(ypred, ytrue)).log().mean()

        ymean, ysd = y.mean(axis = 0).unsqueeze(0), y.std(axis = 0).unsqueeze(0)
        xmean, xsd = x.mean(axis = 0).unsqueeze(0), x.std(axis = 0).unsqueeze(0)
        xdata = (x-xmean)/xsd
        ydata = (y-ymean)/ysd

        self.train(xdata, ydata, loss = loss, ntrain = len(xdata), **train_args)
        self.set_h(self.h.detach().cpu().numpy()*ysd.cpu().numpy())
        self.yscale = lambda u: u*ysd + ymean
        self.xscale = lambda xv: (xv-xmean)/xsd
        self.extras['ex_ymean'] = ymean.cpu().numpy()
        self.extras['ex_xmean'] = xmean.cpu().numpy()
        self.extras['ex_xsd'] = xsd.cpu().numpy()
        self.extras['ex_ysd'] = ysd.cpu().numpy()


def W1(A, B):
    import ot
    a, q = A.shape
    b, q = B.shape

    D = np.linalg.norm(A.reshape(a, 1, q) - B.reshape(1, b, q), axis = -1)
    return ot.emd2(np.ones(a) / a, np.ones(b) / b, D)

def avgW1error(y_true, y_sampled):
    w1s = [W1(y_true[i], y_sampled[i]) for i in range(len(y_true))]
    return np.mean(w1s)