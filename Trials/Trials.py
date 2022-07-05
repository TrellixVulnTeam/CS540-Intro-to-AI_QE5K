from collections import namedtuple
import time
import numpy as np
import copy

# list = [1]
# print(list[:123])
# print(1 // 3)
#
# last_b = None
#
#
# def divide(t, b=None):
#     global last_b
#     if b == None:
#         b = last_b
#     last_b = b
#     return t / b
#
#
# print(divide(1, 4))
# print(divide(2))

# def func():
#     global x
#     print(x)
#     x = 1
#
#
# func()
# print(x)

# print(-4 // 3)
# print(-4 % 3)
# print("123456"[-1])
# print(10 / 5)
# print(5 ^ 2)
# print("12345"[-1:])

# temp = []
# temp += "asdasdas"
# temp += "123123"
# print(temp)

# msg = "This week was a lot of work. At least I finished quiz 5!"
# print(msg[29:])

# a = set()
# a.add(1)
# a.add(2)
# a.add(14)
# print(a)
# msg = "asd"
# print(msg.split(","))
#
# a = np.array([[1, 2, 3], [2, 34, 2]])
# s = np.array([2, 2, 4] * a)
# print(s.sum(axis=1))

# import copy
# a = [1,2,3,4]
# b = a
# c = copy.copy(b)
# a[1] = 1
# a = copy.deepcopy(a)
# a[1] = 3
# d = copy.copy(a)
# print(c[0] == b[0])

# print(sorted([(1,2), (1,1)]))

# print(max((1, 3), (1, 4), ))

# with open("nums.txt") as f:
#     print(list(f))
#     print(list(f))

# class Dog:
#     bool = [True]
#
#     def fu(self):
#         self.bool.append("e")
#
# a1 = Dog()
# a1.fu()
# a2 = Dog()
# print(a1.bool)
# print(a2.bool)

str = """
<table>
    <tr>
        <td><a href="Alabama.html" title="Alabama">Alabama</a></td>
        <td><i>Montgomery</i></td>
    </tr>
    <tr>
        <td><a href="Alaska.html" title="Alaska">Alaska</a></td>
        <td><b>Juneau</b></td>
    </tr>
  <tr> 
        <td><a href="Wisconsin.html" title="Wisconsin">Wisconsin</a></td> 
        <td><i>Madison</i></td> 
  </tr>
  <tr> 
        <td><a title="Wyoming">Wyoming</a></td> 
        <td><b>Cheyenne</b></td> 
  </tr>
</table>
"""

import requests


# from bs4 import BeautifulSoup
# doc = BeautifulSoup(str, "html.parser")
# ss = doc.find_all("tr")[0].children
# print(next(ss))
# print(next(ss))

# li = [{"a": "asd"}]
# for l in li:
#     l = copy.copy(l)
#     l["a"] = 3
# print(li)

class Father:
    def __init__(self):
        self.pos = 0


class Son(Father):
    def __init__(self):
        super().__init__()
        pass


a = Son()


# print(a.pos)

# def f(*args):
#     print(len(args))
#
#
# nums = [1, 2, 3]
# f(*nums)

def f(fn):
    print("A")
    def wrapper(args):
        print(args)
    return wrapper

@f
def g(x):
    print("C")

g(1)