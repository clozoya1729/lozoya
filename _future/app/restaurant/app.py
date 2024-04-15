FRESHNESS_CRITERIA = 2


class Item:
    def __init__(self, cost, name, quantity, minimumQuantity, unitsPerPurchase):
        self.cost = cost
        self.name = name
        self.quantity = quantity
        self.minimumQuantity = minimumQuantity
        self.unitsPerPurchase = unitsPerPurchase

    @property
    def unit_cost(self):
        return self.cost / self.unitsPerPurchase

    def purchase(self):
        self.quantity += self.unitsPerPurchase


class Material(Item):
    def __init__(self, *args, **kwargs):
        super(Material, self).__init__(*args, **kwargs)


class Perishable(Item):
    def __init__(self, freshness, *args, **kwargs):
        super(Perishable, self).__init__(*args, **kwargs)
        self.freshness = freshness
        self.age = freshness

    def rot(self):
        self.freshness -= 1

    @property
    def expired(self):
        return self.freshness <= FRESHNESS_CRITERIA

    def purchase(self):
        super().purchase()
        self.age = self.freshness


class MenuItem:
    def __init__(self, price):
        self.price = price
        self.materials = []


class Salad(MenuItem):
    def __init__(self, *args, **kwargs):
        super(Salad, self).__init__(*args, **kwargs)
        self.materials = [
            ('napkin', 2),
            ('fork', 1),
            ('container', 1),
        ]


class Juice(MenuItem):
    def __init__(self, *args, **kwargs):
        super(Juice, self).__init__(*args, **kwargs)
        self.materials = [
            ('napkin', 2),
            ('straw', 1),
            ('pouch', 1),
        ]


class Restaurant:
    def __init__(self, initialMoney):
        self.money = initialMoney
        self.spend = 0
        self.inventory = {
            'napkin':    Material(
                name="napkin",
                cost=20,
                minimumQuantity=50,
                quantity=0,
                unitsPerPurchase=4000,
            ),
            'straw':     Material(
                name="straw",
                cost=35,
                quantity=0,
                minimumQuantity=50,
                unitsPerPurchase=200,
            ),
            'pouch':     Material(
                name="pouch",
                cost=35,
                quantity=0,
                minimumQuantity=50,
                unitsPerPurchase=100,
            ),
            'fork':      Material(
                name="fork",
                cost=5,
                quantity=0,
                minimumQuantity=50,
                unitsPerPurchase=100,
            ),
            'container': Material(
                name="container",
                cost=55,
                quantity=0,
                minimumQuantity=50,
                unitsPerPurchase=450
            ),
        }
        self.menu = {
            'salad': Salad(
                price=5
            ),
            'juice': Juice(
                price=5
            ),
        }

    def sell(self, item):
        canSell = True
        for material, quantity in self.menu[item].materials:
            if quantity > self.inventory[material].quantity:
                print(material, quantity, self.inventory[material].quantity)
                canSell = False
                print(f'INSUFFICIENT QUANTITY {item}\n\t{material}:\t{quantity}')
        if canSell:
            self.money += self.menu[item].price
            for material, quantity in self.menu[item].materials:
                self.inventory[material].quantity -= quantity
                # print(f'SOLD {item}\n\t{material}:\t{quantity}')

    def purchase(self, item):
        cost = self.inventory[item].cost
        if self.money >= cost:
            self.money -= cost
            self.spend += cost
            self.inventory[item].purchase()
            # print(f'PURCHASE\n\t{item} quantity:\t{self.inventory[item].quantity}')
            return True
        else:
            print(f"${self.money} NOT ENOUGH MONEY TO BUY {item}")
            return False

    def routine_daily(self, day):
        for item in self.inventory:
            itemObject = self.inventory[item]
            while itemObject.quantity <= itemObject.minimumQuantity:
                purchased = self.purchase(item)
                if purchased == False:
                    print('B R O K E')
                    break
            # if type(itemObject) == Material:
            #     pass
            # if type(itemObject) == Perishable:
            #     itemObject.rot()
            #     if itemObject.expired:
            #         self.purchase(item)

    def routine_work(self, day, sales):
        self.routine_daily(day)
        self.money -= 24
        self.spend += 24
        for menuItem in self.menu:
            for i in range(sales):
                self.sell(menuItem)

    def routine_nonwork(self, day):
        self.routine_daily(day)
