
import numpy as np
def _kernel(xj, xi, sigma):
    l2normsquared = np.sqrt(np.sum(np.absolute((xj-xi)**2)))
    sigmasquared = sigma ** 2
    dist = np.exp(-1 * l2normsquared/(2 * sigmasquared))
    if(np.isnan(dist)):
        return 0
    return dist
#     return (xi-xj).sum()

def _get_num_classes(targets):
    counter = 0
    targetmap = {}
    for i in range(len(targets)):
        value = targets[i]
        if value not in targetmap:
            targetmap[value] = True
    return len(targetmap)


def separate(data, targets, sigma):
    """Return an array of datapoints in lower dimension

    Separates data to maximize in-class and within-class variance with free parameter sigma
    uses RBF kernel by default, support for other kernels will be added soon
    """
    #sep data into classes
    num_classes = _get_num_classes(targets)
    print("num classes= " , num_classes)
    dimensions = data.shape[1]
    data_sep_class = []
    for i in range(num_classes):
        data_sep_class.append([])
    
    j = 0
    for i in range(len(data)):
        for j in range(num_classes):
            if(j == targets[i]):
                data_sep_class[j].append(np.asarray(data[i]))
                break
    data_sep_class = np.asarray(data_sep_class)

    print(data_sep_class.shape)
    #Create M matrix by first making the shape
    Ms = []
    for k in range(num_classes):
        M1 = np.zeros((len(data)))
        Ms.append(M1)
    Mstar = Ms

    #Get all the M's
    for k in range(len(Ms)):
        for j in range(len(data)):
            summation = 0
            xj = data[j]
            for xi in data_sep_class[k]:
                summation += _kernel(xi, xj, sigma)
            summation /= len(data_sep_class[k])
            Ms[k][j] = summation #erroring
    
    #Get Mstar, or the mean
    for k in range(len(Ms)):
        for j in range(len(data)):
            summation = 0
            xj = data[j]
            for xi in data_sep_class[k]:
                summation += _kernel(xi, xj, sigma)
            Mstar[0][j] += summation
    Mstar = np.asarray(Mstar)
    Mstar /= num_classes
    
    #Get our final summation M
    finalM = np.zeros((len(data), len(data)))
    for a in range(num_classes):
        i = data_sep_class[a]
        m = Ms[a]
        finalM += len(i) * (m - Mstar).T @ (m-Mstar)              
    M = finalM
    
    #Now we find the K values
    Ks = []
    sizes = []
    for i in range(num_classes):
        K1 = np.zeros((len(data), len(data_sep_class[i])))
        Ks.append(K1)
        sizes.append(len(data_sep_class[i])) 

    for k in range(num_classes):
        Ki = Ks[k]
        for j in range(len(data)):
            xj = data[j]
            for i in range(len(data_sep_class[k][i])):
                xi = data_sep_class[k][i]
                Ki[j][i] = _kernel(xi, xj, sigma)
    
    N = np.zeros((K1 @ K1.T).shape)

    #Now we add them together according to our equation
    for i in range(num_classes):
        N += Ks[i] @ (np.identity(sizes[i]) - 1/sizes[i] * np.ones((sizes[i], sizes[i])) )  @ Ks[i].T
    
    #We perturb the matrix by a small amount to make it nonsingular
    e = .1
    a = np.linalg.inv(N + np.identity(N.shape[0])*e) @ M

    #Now we find eigen values of this matrix
    w, v = np.linalg.eig(a)
    a = v[:num_classes-1] #Top C-1 leading eigenvectors
    Y = []
    
    #Project the data onto this new basis and return the new dataset
    for i in range(len(data)):
        temp = 0
        for j in range(len(data)):
            alpha = a.T[j]
            diff = _kernel(data[i], data[j], sigma)
            temp += alpha * diff
        Y.append(temp)
    data = np.asarray(Y)
    return data
