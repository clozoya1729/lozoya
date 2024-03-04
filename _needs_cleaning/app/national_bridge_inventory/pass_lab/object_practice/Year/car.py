'''from State import Mercedes
class DataStore(Mercedes.Mixin): # Could inherit many more mixins

    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        self.small_method()

    def small_method(self):
        Mercedes.Mixin()
        return self.a'''

'''one, two, three, four = 0, 0, 0, 0
with open("C:\\Users\clozo_000\PycharmProjects\Object Practice\Year\Bridge Item Info - Copy\Item Digits.txt", "r") as file:
    for line in file:
        x = (line.strip())
        if x != '(Reserved)':
            x = int(line.strip())
            print(x)
            if x == 1:
                one += 1
            if x == 2:
                two += 2

print("One digit: " + str(one))
print("Two digits: " + str(two))'''

with open("C:\\Users\clozo_000\PycharmProjects\Object Practice\Year\Bridge Item Info - Copy\Item Number.txt", "r") as file:
    i = 0
    for line in file:
        #if i == 0:
        with open("C:\\Users\clozo_000\PycharmProjects\Object Practice\Year\Bridge Item Info - Copy\\Item " + line.strip() + ".txt", "w") as new:
            pass#i = 1
