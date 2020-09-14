import random
import numbers
import math
import pickle
import copy

def product(array):
    if isinstance(array, Tensor):
        array = array.array
    result = 1
    for i in array:
        result *= i
    return result

class Tensor():

    """
    dims in order for 2D tensor/matrix:rows/height,columns/width
    dims in order for 3D tensor:layers/depth,rows/height,columns/width
    for an ND tensor follow this pattern for declaration of more spacial
    dimensions.
    """
    def __init__(self, dims=False, array=False):
        if isinstance(array, list):
            self.array = array
            self.dims = Tensor.recursive_find_dims(self.array, [], True)
        elif isinstance(dims, list) and isinstance(dims[0], int):
            self.dims = dims
            self.array = Tensor.recursive_build(self.dims)
        else:
            raise AttributeError("""dimensions or an array must be
                                 provided to initialisation""")

    def max_norm_constrain(array, max_value=4, first=True):
        r = []
        if first:
            if isinstance(array, Tensor):
                array = array.array
        if isinstance(array, list):
            for a in array:
                r.append(Tensor.max_norm_constrain(a, max_value, False))
        else:
            r = max(-4, min(4, array))
        if first:
            return Tensor(array=r)
        return r
        
    def generate_zero_mask(dims, percent=30, first=True):
        r = []
        if len(dims)>1:
            for _ in range(dims[0]):
                r.append(Tensor.generate_zero_mask(dims[1:], percent, False))
        else:
            for _ in range(dims[0]):
                r.append(1 if random.randrange(0, 100)>=percent else 0)
        if first:
            return Tensor(array=r)
        return r

    def weight_statistics(array, lower=-4, upper=4, step=1, r=False):
        first = not r
        if first:
            r = {}
            if isinstance(array, Tensor):
                array = array.array
            for i in range(lower, upper+1, step):
                r[i] = 0
        if isinstance(array, list):
            for a in array:
                Tensor.weight_statistics(a, lower, upper, step, r)
        else:
            r[min(r, key = lambda x: abs(x-array))] += 1 
        if first:
            return r
        

    def recursive_find_dims(array, dims, first=False):
        dims.append(len(array))
        if isinstance(array[0], list):
            Tensor.recursive_find_dims(array[0], dims)
        if first:
            return dims

    def recursive_build(dims):
        array = []
        if len(dims)>1:
            for _ in range(dims[0]):
                array.append(Tensor.recursive_build(dims[1:]))
        else:
            for _ in range(dims[0]):
                array.append(random.uniform(-1, 1))
        return array

    def recursive_str(array):
        string = ""
        for i in array:
            if isinstance(array[0], list):
                string += Tensor.recursive_str(i)+"\n"
            else:
                t = str(i)
                if len(t)<3:
                    t += " "*(3-len(t))
                elif len(t)>3:
                    t = t
                string += "{0} ".format(t)
        return string

    def __str__(self):
        return Tensor.recursive_str(self.array)

    def __repr__(self):
        return "Tensor of dimensions: {0}".format(self.dims)

    def recursive_math(array, other, func, start=False):
        if isinstance(array, Tensor):
            array.check_dims(other)
            array = array.array
        other = other if not isinstance(other, Tensor) else other.array
        to_return = []
        if isinstance(array[0], list):
            if isinstance(other, list):
                for i, j in zip(array, other):
                    to_return.append(Tensor.recursive_math(i, j, func))
            else:
                for i in array:
                    to_return.append(Tensor.recursive_math(i, other, func))
        else:
            if isinstance(other, list):
                for i, j in zip(array, other):
                    to_return.append(func(i, j))
            else:
                for i in array:
                    to_return.append(func(i, other))
        if start:
            return Tensor(array=to_return)
        return to_return

    def __add__(self, other):
        return Tensor.recursive_math(self, other, lambda x,y:x+y, True)

    def __radd__(self, other):
        return Tensor.__add__(self, other)

    def __sub__(self, other):
        return Tensor.recursive_math(self, other, lambda x,y:x-y, True)

    def __rsub__(self, other):
        return Tensor.__sub__(self, other)

    def __mul__(self, other):
        return Tensor.recursive_math(self, other, lambda x,y:x*y, True)

    def __rmul__(self, other):
        return Tensor.__mul__(self, other)

    def __div__(self, other):
        return Tensor.recursive_math(self, other, lambda x,y:x/y, True)

    def __rdiv__(self, other):
        return Tensor.__mul__(self, other)

    def recursive_relu(array, deriv=False, leaky=0):
        r = []
        if isinstance(array[0], list):
            for a in array:
                r.append(Tensor.recursive_relu(a))
        elif deriv:
            for a in array:
                r.append(1 if a>0 else leaky)
        else:
            for a in array:
                r.append(a if a>0 else leaky*a)
        return r

    def relu(self, deriv=False):
        return Tensor(array=Tensor.recursive_relu(self.array, deriv))
            
    def leaky_relu(self, deriv=False, leaky=-0.1):
        return Tensor(array=Tensor.recursive_relu(self.array, deriv, leaky))

    def check_dims(self, other):
        t = "Dimension Error: Tensors of dimensions {0} and {1} don't match"
        if isinstance(other, Tensor):
            if other.dims != self.dims:
                raise Exception(t.format(self.dims, other.dims))

    def sum(array):
        result = 0
        if isinstance(array[0], list):
            for i in array:
                result += Tensor.sum(i)
        else:
            for i in array:
                result += i
        return result

    def recursive_dot():
        pass

    def dot(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Argument to dot product must be a Tensor object.")
        if len(other.dims)!=2:
            raise ValueError("Dot only implemented for 2D Tensor argument.")
        if self.dims[-1] != other.dims[-2]:
            t = "{0}\nself: {1} other: {2}"
            t1 = "Rows and Columns of matrices do not align"
            t = t.format(t1, self.dims, other.dims)
            raise ValueError(t)
        x = []
        for a in range(self.dims[-2]):
            y = []
            for b in range(other.dims[-1]):
                z = 0
                for c in range(self.dims[-1]):
                    z += self.array[a][c]*other.array[c][b]
                y.append(z)
            x.append(y)
        return Tensor(array=x)

    def recursive_polarise(array):
        if isinstance(array, list):
            if isinstance(array[0], list):
                a = []
                for i in array:
                    a.append(Tensor.recursive_polarise(i))
                return a
            else:
                a = []
                for i in array:
                    a.append(-1 if i<=0 else 1)
                return a

    def rec_dummy(array):
        r = []
        if isinstance(array, list):
            for a in array:
                r.append(Tensor.rec_dummy(a))
        else:
            return array
        return r

    def dummy(self, *args, **kwargs):
        return Tensor(array=Tensor.rec_dummy(self.array))

    def polarise(self):
        array = Tensor.recursive_polarise(self.array)
        self.array = array

    def recursive_flatten(self, array):
        for i in array:
            if isinstance(array[0], list):
                self.recursive_flatten(i)
            else:
                self.flattened.append(i)
                
    def flatten(self):
        self.flattened = []
        self.recursive_flatten(self.array)
        return Tensor(array=self.flattened)

    def recursive_max_pool(array, window_size=2, stride=2):
        a = []
        if not isinstance(array[0][0], list):
            for i in range(len(array)//stride):
                b = []
                i *= stride
                for j in range(len(array[0])//stride):
                    j *= stride
                    v = -99999999999999
                    for k in range(window_size):
                        for l1 in range(window_size):
                            try:
                                if i+k>=0 and j+l1>=0:
                                    v = max(v, array[i+k][j+l1])
                            except IndexError:
                                pass
                    b.append(v)
                a.append(b)
        else:
            for ar in array:
                a.append(Tensor.recursive_max_pool(ar, window_size, stride))
        return a                

    def make_max_pool_mask(self, array, window_size, stride, first=False):
        a = []
        if not isinstance(array[0][0], list):
            a = []
            for _ in array[0]:
                b = []
                for _ in array:
                    b.append(0)
                a.append(b)
            for i in range(len(array)//stride):
                i *= stride
                for j in range(len(array[0])//stride):
                    j *= stride
                    v = -99999999999999
                    for k in range(window_size):
                        for l1 in range(window_size):
                            try:
                                if i+k>=0 and j+l1>=0:
                                    v = max(v, array[i+k][j+l1])
                            except IndexError:
                                pass
                    for k in range(window_size):
                        for l1 in range(window_size):
                            try:
                                if i+k>=0 and j+l1>=0:
                                    if array[i+k][j+l1] >= v and v>0:
                                        a[i+k][j+l1] = 1
                            except IndexError:
                                pass
        else:
            for ar in array:
                a.append(self.make_max_pool_mask(ar, window_size, stride))
        if first:
            return Tensor(array=a).reshape(self.dims)
        return a

    def max_pool(self, window_size=2, stride=2):
        mask = self.make_max_pool_mask(self.array, window_size, stride, True)
        pooled = Tensor(array=Tensor.recursive_max_pool(self.array,
                                                        window_size, stride))
        return (pooled, mask)

    def recursive_reverse_max_pool(array, mask, window_size,
                                   stride, first=False):
        a = []
        if not isinstance(array[0][0], list):
            for i in range(len(mask)):
                b = []
                for j in range(len(mask[0])):
                    b.append(array[i//stride][j//stride]*mask[i][j])
                a.append(b)
        else:
            for ar, ms in zip(array, mask):              
                a.append(Tensor.recursive_reverse_max_pool(ar, ms, window_size,
                                                          stride))
        if first:
            return Tensor(array=a)
        return a
            

    def reverse_max_pool(self, mask, window_size=2, stride=2):
        return Tensor.recursive_reverse_max_pool(self.array, mask.array,
                                                 window_size, stride, True)
        
    def recursive_reshape(self, out_array, counter, in_array, dims):
        if counter+1==len(dims):
            for i in range(dims[counter]):
                out_array.append(in_array[self.counter])
                self.counter += 1
        else:
            for i in range(dims[counter]):
                out_array.append(self.recursive_reshape([], counter+1,
                                                        in_array, dims))
        return out_array

    def reshape(self, dims):
        self.counter = 0
        if product(self.dims)!=product(dims):
            raise ValueError("Tensor will not fit into specified dimensions.")
        in_array = self.flatten().array
        return Tensor(array=self.recursive_reshape([], 0, in_array, dims))

    def convolve_dims(image, kernel, stride):
        return image[:-2]+[kernel[0], image[-2]//stride, image[-1]//stride]

    def rec_reverse_con(a, a0, fs, stride, pl, pr, pt, pb):
        u = []
        if isinstance(a[0][0], list) and isinstance(a0[0][0], list):
            for t, t0 in zip(a, a0):
                u.append(Tensor.rec_reverse_con(t, t0, fs, stride, pl,
                                                pr, pt, pb))
        elif isinstance(a[0][0], list):
            for t in a:
                u.append(Tensor.rec_reverse_con(t, a0, fs, stride,
                                                pl, pr, pt, pb))
        elif isinstance(a0[0][0], list):
            for t in a0:
                u.append(Tensor.rec_reverse_con(a, t, fs, stride,
                                                pl, pr, pt, pb))
        else:
            for _ in range(fs):
                u0 = []
                for _ in range(fs):
                    u0.append(0)
                u.append(u0)
            for i in range(-pt, len(a[0])-pb):
                for j in range(-pl, len(a[0])-pr):
                    for l in range(fs):
                        for m in range(fs):
                            try:
                                if i+l>0 and j+m>0:
                                    u[l][m] += a[i+l][j+m]*a0[i][j]
                            except IndexError:
                                pass
        return u

    def merge(self):
        x = Tensor(array=self.array[0])
        for i in self.array[1:]:
            x += Tensor(array=i)
        x = x.__div__(len(self.array))
        return x

    def reverse_convolve(self, other, fs=3, l=False, r=False,
                         t=False, b=False, stride=1, pad=False):
        if not isinstance(other, Tensor):
            raise TypeError("""Argument to convolution operation
                                    must be a Tensor object.""")
        if pad:
            l, r, t, b = pad, pad, pad, pad
        l = 1
        r = 1
        t = 1
        b = 1
        asd =  Tensor(array=Tensor.rec_reverse_con(self.array, other.array, fs, stride, l, r, t, b))
        asd = asd.merge()
        return asd
        

    def rec_con(a1, a2, stride, pl, pr, pt, pb):
        r = []
        m = []
        if isinstance(a1[0][0], list) and isinstance(a2[0][0], list):
            for t1 in a1:
                r1 = []
                m0 = []
                for t2 in a2:
                    af, mf = Tensor.rec_con(t1, t2, stride, pl, pr, pt, pb)
                    r1.append(af)
                    m0.append(mf)
                r.append(r1)
                m.append(m0)
        elif isinstance(a1[0][0], list):
            for t1 in a1:
                af, mf = Tensor.rec_con(t1, a2, stride, pl, pr, pt, pb)
                r.append(af)
                m.append(mf)
        elif isinstance(a2[0][0], list):
            for t2 in a2:
                af, mf = Tensor.rec_con(a1, t2, stride, pl, pr, pt, pb)
                r.append(af)
                m.append(mf)
        else:
            output_rows = len(a1)//stride-len(a2[0])+1
            output_columns = len(a1[0])//stride-len(a2)+1
            for i in range(-pt, output_rows+pb):
                i *= stride
                e = []
                m0 = []
                for j in range(-pl, output_columns+pr):
                    j *= stride
                    x = 0
                    m1 = []
                    for a in range(len(a2)):
                        m2 = []
                        for b in range(len(a2[0])):
                            try:
                                if a+i>=0 and b+j>=0:
                                    v = a1[a+i][b+j]*a2[a][b]
                                    m2.append(v)
                                    x += v
                                else:
                                    m2.append(0)
                            except IndexError:
                                m2.append(0)
                                pass
                        m1.append(m2)
                    m0.append([[y/x if x else 0 for y in z] for z in m1])
                    e.append(x)
                m.append(m0)
                r.append(e)
        return r, m

    def convolve(self, other, l=False, r=False, t=False, b=False,
                 stride=1, pad=False):
        if not isinstance(other, Tensor):
            raise TypeError("""Argument to convolution operation
                                    must be a Tensor object.""")
        if pad:
            l, r, t, b = pad, pad, pad, pad
        l = (other.dims[-1]-1)//2 if not l else l
        r = (other.dims[-1]-1)//2+(other.dims[-1]-1)%2 if not r else r
        t = (other.dims[-2]-1)//2 if not t else t
        b = (other.dims[-2]-1)//2+(other.dims[-2]-1)%2 if not b else b
        array, mask = Tensor.rec_con(self.array, other.array, stride, l, r,t,b)
        return Tensor(array=array), Tensor(array=mask)
        

    def rec_tanh(array, deriv):
        r = []
        if isinstance(array, list):
            for a in array:
                r.append(Tensor.rec_tanh(a, deriv))
        elif deriv:
            r = 1-(math.tanh(array)**2)
        else:
            r = math.tanh(array)
        return r

    def tanh(self, deriv=False):
        return Tensor(array=Tensor.rec_tanh(self.array, deriv))

    def rec_soft_max(array, deriv):
        r = []
        if isinstance(array[0], list):
            for a in array:
                r.append(Tensor.rec_soft_max(a, deriv))
        elif deriv:
            raise Exception("FFS")
        else:
            s = 0
            for a in array:
                try:
                    s += math.e**a
                except Exception as e:
                    print(e)
                    print(s)
                    print(a)
            for a in array:
                try:
                    r.append(math.e**a/s)
                except ZeroDivisionError:
                    r.append(0)
                except Exception as e:
                    print(e)
                    print(s)
                    print(a)
        return r

    def soft_max(self, deriv=False):
        return Tensor(array=Tensor.rec_soft_max(self.array, deriv))
        
    def sigmoid(self, deriv=False):
        return Tensor(array=Tensor.recursive_sigmoid(self.array, deriv))

    def recursive_sigmoid(array, deriv):
        r = []
        if isinstance(array, list):
            for i in array:
                r.append(Tensor.recursive_sigmoid(i, deriv))
        elif deriv:
            r = (1/(1+math.exp(-array)))*(1-(1/(1+math.exp(-array))))
        else:
            try:
                r = 1/(1+math.exp(-array))
            except:
                r = 0
        return r

    def rec_transpose(array, first=False):
        result = []
        if isinstance(array, list):
            if isinstance(array[0], list):
                if isinstance(array[0][0], list):
                    for sub_array in array:
                        result.append(Tensor.rec_transpose(sub_array[index]))
                else:
                    for index in range(len(array[0])):
                        result.append([])
                        for sub_array in array:
                            result[index].append(sub_array[index])
        if first:
            return Tensor(array=result)
        return result

    def transpose(self):
        return Tensor.rec_transpose(self.array, True)

    def recursive_one_hot(array, first=False):
        result = []
        if isinstance(array, list):
            if isinstance(array[0], list):
                for sub_array in array:
                    result.append(Tensor.recursive_one_hot(sub_array))
            else:
                max_val = -99999999999999
                for val in array:
                    max_val = val if val>max_val else max_val
                for val in array:
                    result.append(1 if val>=max_val else 0)
                    max_val += 1 if val>=max_val else 0
        if first:
            return Tensor(array=result)
        return result

    def one_hot(self):
        return Tensor.recursive_one_hot(self.array, True)

    def recursive_add_weights(array):
        if isinstance(array[0], list):
            for i in array:
                Tensor.recursive_add_weights(i)
        else:
            array.append(0)
    
    def add_weights(self):
        Tensor.recursive_add_weights(self.array)
        self.dims[-1] += 1

if __name__=="__main__":
    import Network
    t = Tensor.generate_zero_mask([10, 28, 28])
    print(t)




    









        



