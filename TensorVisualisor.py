import tkinter as Tk
import pickle
from ImmutableEntry import ImmutableEntry
from Tensor import Tensor
from Network import *

def dummy():
    pass

def rgb(r, g, b):
    r = str(hex(max(min(int(r), 255), 0)))[2:]
    g = str(hex(max(min(int(g), 255), 0)))[2:]
    b = str(hex(max(min(int(b), 255), 0)))[2:]
    r = r if len(r)==2 else r+"0"
    g = g if len(g)==2 else g+"0"
    b = b if len(b)==2 else b+"0"
    return "#{0}{1}{2}".format(r, g, b)


class MyFrame(Tk.Frame):

    def clear(self):
        for widget in self.winfo_children():
            widget.destroy()


class TensorVisualisor(MyFrame):

    def __init__(self, parent, settings):
        Tk.Frame.__init__(self, parent)
        self.canvas = Tk.Canvas(self)
        self.canvas.place(relx=0, rely=0.2, relwidth=1, relheight=0.8)
        self.config(borderwidth=4, relief="raised")
        self.dim_buttons = []
        self.linked = []
        self.settings = settings
        self.settings.tensor_visualisors.append(self)

    def destroy(self):
        self.settings.tensor_visualisors.remove(self)
        Tk.Frame.destroy(self)

    def clear(self):
        for widget in self.winfo_children():
            if widget!=self.canvas:
                widget.destroy()
        self.canvas.delete("all")
        
    def display(self, array):
        self.update()
        h = self.canvas.canvasx(self.canvas.winfo_width())//len(array[0])
        w = self.canvas.canvasy(self.canvas.winfo_height())//len(array)
        self.h, self.w = h, w
        for i, a in enumerate(array):
            for j, b in enumerate(a):
                coords = (j*h, i*w, (j+1)*h, (i+1)*w)
                fill = rgb(b*256, 0, -b*256)
                self.canvas.create_rectangle(coords, fill=fill)
                if self.settings.numbers:
                    fill = rgb(0, 255, 255)
                    t = str(b)[:3] if b>0 else str(b)[:4]
                    coords = ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)
                    self.canvas.create_text(coords, fill=fill, text=t)

    def refresh(self):
        self.visualise(self.tensor, self.indices)

    def visualise(self, tensor, indices=False):
        self.tensor = tensor
        self.clear()
        if len(tensor.dims)<3:
            self.display(tensor.array)
        else:
            self.indices = [0]*(len(tensor.dims)-2) if not indices else indices
            to_display = tensor.array
            for i in self.indices:
                to_display = to_display[i]
            self.display(to_display)
            self.display_dimension_options(tensor)
        self.update()

    def link(self, linked, fire_index, call_index):
        self.linked.append([linked, fire_index, call_index])

    def animate_max_pool(self, linked, window_size, position):
        w, p = window_size, position
        self.canvas.create_rectangle(p[0]*self.h, p[1]*self.w,
                                     (p[0]+w)*self.h, (p[1]+w)*self.w,
                                     outline=rgb(0,255,255))
        result = []
        for i in range(window_size):
            row = []
            for j in range(window_size):
                row.append(self.tensor.array[p[0]+i][p[1]+j])
            result.append(row)
        return result

    def move(self, index, increment, linked=False):
        self.indices[index] += increment
        self.indices[index] = min(max(0, self.indices[index]),
                                  self.tensor.dims[index]-1)
        self.visualise(self.tensor, self.indices)
        if not linked:
            for linked, fire_index, call_index in self.linked:
                if index==fire_index:
                    linked.move(call_index, increment, True)

    def display_dimension_options(self, tensor):
        text = "Tensor has {0} spacial dimensions.".format(len(tensor.dims))
        label = Tk.Label(self, text=text)
        label.place(relx=0.02, rely=0.01, relwidth=0.96, relheight=0.06)
        for i, index in enumerate(self.indices):
            c0 = lambda i=i, b=-1: self.move(i, b)
            c1 = lambda i=i, b=1: self.move(i, b)
            t = "{0} dimension".format(2+len(self.indices)-i)
            l0 = Tk.Label(self, text=t)
            b0 = Tk.Button(self, text="-", command=c0)
            l1 = Tk.Label(self, text=str(index))
            b1 = Tk.Button(self, text="+", command=c1)
            l0.place(relx=i*0.2+0.1, rely=0.07, relwidth=0.15, relheight=0.045)
            b0.place(relx=i*0.2+0.1, rely=0.13, relwidth=0.05, relheight=0.06)
            l1.place(relx=i*0.2+0.15, rely=0.13, relwidth=0.05, relheight=0.06)
            b1.place(relx=i*0.2+0.2, rely=0.13, relwidth=0.05, relheight=0.06)
            self.dim_buttons.append([b0, b1])


class ConvolutionVisualisor(MyFrame):

    def __init__(self, parent, settings):
        Tk.Frame.__init__(self, parent, borderwidth=4, relief="raised")
        self.settings = settings

    def link_visualisors(self, img_vis, ker_vis, res_vis):
        ker_vis.link(res_vis, 0, 1)
        res_vis.link(ker_vis, 1, 0)
        for i in range(len(img_vis.tensor.dims)-2):
            img_vis.link(res_vis, i, i)
            res_vis.link(img_vis, i, i)

    def visualise(self, image, kernel, result):
        img_vis = TensorVisualisor(self, self.settings)
        ker_vis = TensorVisualisor(self, self.settings)
        res_vis = TensorVisualisor(self, self.settings)
        img_vis.place(relx=0, rely=0.2, relwidth=0.4, relheight=0.8)
        ker_vis.place(relx=0.4, rely=0.45, relwidth=0.2, relheight=0.4)
        res_vis.place(relx=0.6, rely=0.2, relwidth=0.4, relheight=0.8)
        img_vis.visualise(image)
        ker_vis.visualise(kernel)
        res_vis.visualise(result)
        self.link_visualisors(img_vis, ker_vis, res_vis)
    
class MaxPoolVisualisor(MyFrame):

    def __init__(self, parent, settings):
        Tk.Frame.__init__(self, parent, borderwidth=4, relief="raised")
        self.settings = settings
        b = Tk.Button(self, text="Animate", command=self.animate)
        b.place(relx=0.4, rely=0.05, relwidth=0.2, relheight=0.1)
        self.large = None

    def animate(self):
        if self.large is None:
            return
        array = self.large_vis.animate_max_pool(self.small_vis,
                                                self.window_size, [0,0])
        print(array)

    def link_visualisors(self):
        for i in range(len(self.large_vis.tensor.dims)-2):
            self.large_vis.link(self.small_vis, i, i)
            self.small_vis.link(self.large_vis, i, i)

    def visualise(self, large, small, window_size, stride):
        self.large, self.small = large, small
        self.window_size, self.stride = window_size, stride
        large_vis = TensorVisualisor(self, self.settings)
        small_vis = TensorVisualisor(self, self.settings)
        large_vis.place(relx=0, rely=0.2, relwidth=0.6, relheight=0.8)
        small_vis.place(relx=0.6, rely=0.3, relwidth=0.4, relheight=0.8*(2/3))
        large_vis.visualise(large)
        small_vis.visualise(small)
        self.large_vis, self.small_vis = large_vis, small_vis
        self.link_visualisors()


class NetworkVisualisorSettings(MyFrame):

    def __init__(self, parent):
        Tk.Frame.__init__(self, parent, borderwidth=4, relief="raised")
        self.tensor_visualisors = []
        self.numbers = False
        b = Tk.Button(self, text="Show Numbers", fg="red")
        b.config(command=lambda b=b: self.invert_numbers(b))
        b.place(relx=0, relwidth=0.1, rely=0, relheight=1)

    def invert_numbers(self, b):
        self.numbers = not self.numbers
        b.config(fg="green" if self.numbers else "red")
        for tensor_visualisor in self.tensor_visualisors:
            tensor_visualisor.refresh()


class NetworkVisualisor(Tk.Tk):

    def __init__(self, network=False):
        Tk.Tk.__init__(self)
        self.layer_buttons, self.conv_buttons, self.max_buttons = [], [], []
        self.settings = NetworkVisualisorSettings(self)
        self.settings.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.1)
        self.vis = None
        self.net = network
        self.geometry("1400x850")
        if network:
            self.visualise()

    def display_convolution(self, image, kernel, result):
        if self.vis is not None:
            self.vis.destroy()
        self.vis = ConvolutionVisualisor(self, self.settings)
        self.vis.place(relx=0.02, rely=0.2, relwidth=0.96, relheight=0.8)
        self.vis.visualise(image, kernel, result)

    def display_tensor(self, tensor):
        if self.vis is not None:
            self.vis.destroy()
        self.vis = TensorVisualisor(self, self.settings)
        self.vis.place(relx=0.2, rely=0.2, relwidth=0.6, relheight=0.8)
        self.vis.visualise(tensor)

    def display_layer(self, layer, c, c0):
        if isinstance(layer, list):
            t = "Output" if c0+2==len(self.net.layers) else "Hidden"
            b1 = Tk.Button(self, text=t)
            b1.place(x=10+120*c, rely=0, width=120, relheight=0.05)
            b = Tk.Button(self, text=str(layer))
            b.place(x=10+120*c, rely=0.05, width=120, relheight=0.05)
            self.layer_buttons.append((b, b1))
        else:
            if layer.layer_type == "max_pool":
                com = dummy
            else:
                com = lambda tensor=layer.tensor: self.display_tensor(tensor)
            test = layer.layer_type != "max_pool"
            t = str(layer.tensor.dims) if test else "N/A"
            b = Tk.Button(self, text=t, command=com)
            b1 = Tk.Button(self, text=layer.layer_type)
            b.place(x=10+120*c, rely=0, width=120, relheight=0.05)
            b1.place(x=10+120*c, rely=0.05, width=120, relheight=0.05)
            if layer.layer_type == "convolutional":
                self.conv_buttons.append(b1)
            elif layer.layer_type == "max_pool":
                self.max_buttons.append(b1)
            elif layer.layer_type == "input":
                b1.config(command=lambda t=layer.tensor:self.display_tensor(t))

    def display_max_pool(self, large, small, window_size, stride):
        if self.vis is not None:
            self.vis.destroy()
        self.vis = MaxPoolVisualisor(self, self.settings)
        self.vis.place(relx=0.02, rely=0.2, relwidth=0.96, relheight=0.8)
        self.vis.visualise(large, small, window_size, stride)
        

    def forward_pass(self, tensor):
        self.net.forward_pass(Tensor(array=tensor[:50]))
        it = zip(self.layer_buttons, self.net.pass_results[1:])
        for (b0, b1), tensor in it:
            b0.config(command=lambda x=tensor: self.display_tensor(x))
            b1.config(command=lambda x=tensor: self.display_tensor(x))
        it = zip(self.conv_buttons, self.net.convolutional_layers())
        for button, (index, kernel) in it:
            image = self.net.pass_results[index-1]
            result = self.net.pass_results[index]
            button.config(command = lambda i=image, k=kernel, r=result:
                          self.display_convolution(i, k, r))
        it = zip(self.max_buttons, self.net.max_layers())
        for button, index in it:
            large = self.net.pass_results[index-1]
            small = self.net.pass_results[index]
            window_size = self.net.layers[index].window_size
            stride = self.net.layers[index].stride
            button.config(command = lambda l=large, s=small, w=window_size,
                          st=stride: self.display_max_pool(l, s, w, st))

    def visualise(self, network=False):
        i = 0
        i0 = 0
        self.net = network if network else self.net
        for layer in self.net.layers:
            self.display_layer(layer, i, i0)
            i += 1
            if layer.layer_type != "input":
                self.display_layer(layer.result_dims, i, i0)
                i += 1
                i0 += 1
                
network = Network()
path = "/Users/user/Desktop/Tensor/"
images = pickle.load(open("{0}test_images.pickle".format(path), "rb"))
labels = pickle.load(open("{0}test_labels.pickle".format(path), "rb"))
print("LOADING NETWORK")
network.load(84.6)
print("NETWORK LOADED")

visualisor = NetworkVisualisor(network)
visualisor.forward_pass(images)




