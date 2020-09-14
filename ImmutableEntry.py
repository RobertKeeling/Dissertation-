from tkinter import *

class ImmutableEntry(Entry):

    def __init__(self, parent, text):
        Entry.__init__(self, parent, disabledforeground = "black",
                       justify = "center")
        self.insert(0, text)
        self.config(state = "disabled")
