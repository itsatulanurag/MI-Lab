import numpy as np


class Tensor:
    def __init__(self, arr, requires_grad=True):
        self.arr = arr
        self.requires_grad = requires_grad
        self.history = ['leaf', None, None]
        self.zero_grad()
        self.shape = self.arr.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.arr)

    def set_history(self, op, operand1, operand2):
        self.history = []
        self.history.append(op)
        self.requires_grad = False
        self.history.append(operand1)
        self.history.append(operand2)

        if operand1.requires_grad or operand2.requires_grad:
            self.requires_grad = True

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise ArithmeticError(
                    f"Shape mismatch for +: '{self.shape}' and '{other.shape}' ")
            out = self.arr + other.arr
            out_tensor = Tensor(out)
            out_tensor.set_history('add', self, other)

        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")

        return out_tensor

    def __matmul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"unsupported operand type(s) for matmul: '{self.__class__}' and '{type(other)}'")
        if self.shape[-1] != other.shape[-2]:
            raise ArithmeticError(
                f"Shape mismatch for matmul: '{self.shape}' and '{other.shape}' ")
        out = self.arr @ other.arr
        out_tensor = Tensor(out)
        out_tensor.set_history('matmul', self, other)

        return out_tensor

    def grad_add(self, gradients=None):
        a = self.history[1]
        b = self.history[2]
        a.grad = np.zeros_like(a.arr)
        b.grad = np.zeros_like(b.arr)

        if a.requires_grad:
            a.grad += np.ones_like(a.arr)
        if b.requires_grad:
            b.grad += np.ones_like(b.arr)

        if gradients is None:
            return (a.grad, b.grad)

        if a.requires_grad:
            a.grad = np.multiply(np.ones_like(a.arr), gradients)
        if b.requires_grad:
            b.grad = np.multiply(np.ones_like(b.arr), gradients)

        return (a.grad, b.grad)

    def grad_matmul(self, gradients=None):
        a = self.history[1]
        b = self.history[2]

        if gradients == None:
            if a.requires_grad:
                a.grad += np.matmul(np.ones_like(a.arr), b.arr.transpose())
            if b.requires_grad:
                b.grad += np.matmul(np.ones_like(b.arr), a.arr).transpose()

        else:
            if a.requires_grad:
                a.grad += np.multiply(np.matmul(np.ones_like(a.arr),
                                      b.arr.transpose()), gradients)
            if b.requires_grad:
                b.grad += np.multiply(np.matmul(np.ones_like(b.arr),
                                      a.arr).transpose(), gradients)

        return(a.grad, b.grad)

    def backward(self, gradients=None):
        if self.requires_grad == None:
            return

        if self.history[0] == 'add':
            gradient = self.grad_add(gradients)
            if self.history[1]:
                self.history[1].backward(gradient[0])
            if self.history[2]:
                self.history[2].backward(gradient[1])

        elif self.history[0] == 'matmul':
            gradient = self.grad_matmul(gradients)
            if self.history[1]:
                self.history[1].backward(gradient[0])
            if self.history[2]:
                self.history[2].backward(gradient[1])

        else:
            if self.requires_grad:
                self.grad = gradients
