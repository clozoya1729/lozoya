class Plant():
    def __init__(
        self, witherVal=20, dryVal=5, optimalMoisture=50,
        maturity=0, moistureRange=10,
        waterHealthBoost=5, waterMoistureBoost=20,
        lightHealthBoost=10, saturation=100,
        health=100, dank=False, dead=False, name='Plant'
    ):
        self.witherVal = witherVal
        self.dryVal = dryVal
        self.optimalMoisture = optimalMoisture
        self.moistureRange = moistureRange
        self.waterHealthBoost = waterHealthBoost
        self.waterMoistureBoost = waterMoistureBoost
        self.lightHealthBoost = lightHealthBoost
        self.saturation = saturation
        self.health = health
        self.maturity = maturity
        m = optimalMoisture
        if m >= saturation:
            self.moisture = saturation
        else:
            self.moisture = m
        self.dank = dank
        self.dead = dead
        self.name = name

    def __repr__(self):
        return '{}\nHP: {}\nM: {}\n'.format(
            self.name,
            self.health,
            self.moisture
        )

    def rename(self, new):
        self.name = new

    def water(self):
        new = self.health + self.waterHealthBoost
        if new > 100:
            self.health = 100
        else:
            self.health = new
        new = self.moisture + self.waterMoistureBoost
        if new > self.saturation:
            self.moisture = self.saturation
        else:
            self.moisture = new

    def dry(self):
        new = self.moisture - self.dryVal
        if new >= 0:
            self.moisture = new
        else:
            self.moisture = 0

    def wither(self):
        new = self.health - self.witherVal
        if new <= 0:

            self.health = 0
            self.die()
        else:
            self.health = new

    def photosynthesis(self):
        new = self.health + self.lightHealthBoost
        if new > 100:
            self.health = 100
        else:
            self.health = new

    def die(self):
        self.dead = True

    def harvest(self):
        if self.mature == True:
            self.harvested = True
            return True
        print("This plant is not yet mature!\nMaturity: {}".format(self.maturity))
        return False

    def increase_maturity(self):
        new = self.maturity + 40
        if new < 100:
            self.maturity = new
        else:
            self.maturity = 100
        if self.maturity == 100:
            self.mature = True


def buy_item(item, quantity, inventory):
    inventory[item] += quantity
    inventory['money'] -= prices[item] * quantity
    return inventory


def check_inventory(inventory):
    for item in inventory:
        print('{}: {}'.format(item, inventory[item]))


def harvest_crop(plant, inventory):
    if plant.havest() == True:
        inventory['flowers'] += flowersPerPlant
    return inventory


def sell_item(item, quantity, inventory):
    if inventory[item] > 0:
        inventory[item] -= quantity
        inventory['money'] += prices[item] * quantity
    return inventory


def plant_seed(inventory):
    if (inventory['beds'] == 0):
        print('No space in this bed.')
    else:
        if inventory['seeds'] > 0:
            inventory['plants'].append(Plant())
            inventory['seeds'] -= 1
            inventory['plants'][-1].name += ' {}'.format(len(inventory['plants']))
            msg = 'Do you want to rename {}?'.format(inventory['plants'][-1].name)
            selection = input(msg)
            while (selection.lower() != 'no' and selection.lower() != 'yes'):
                msg = 'Say Yes or No.'
                selection = input(msg)
            if selection.lower() == 'yes':
                msg = 'What do you want to name this plant?'
                original = False
                while original == False:
                    newName = input(msg)
                    original = True
                    for plant in inventory['plants']:
                        if plant.name == newName:
                            original = False
                            msg = 'That name already exists. Select a new name.'
                inventory['plants'][-1].name = newName
    return inventory


def water_plant(inventory):
    msg = 'Which plant?'
    found = False
    while found == False:
        plantName = input(msg)
        for plant in inventory['plants']:
            if plantName == plant.name:
                found = True
        msg = 'That plant does not exist.'
    plantIndex = get_plant_index(inventory, plantName)
    alive = not inventory['plants'][plantIndex].dead
    while alive == False:
        plantName = input(msg)
        for plant in inventory['plants']:
            if plant.dead == False:
                alive = True
        msg = 'That plant is dead.'
    plantIndex = get_plant_index(inventory, plantName)
    plant = inventory['plants'][plantIndex]
    old = plant.moisture
    plant.water()

    name = plant.name
    new = plant.moisture
    print('{} has been watered. Moisture went from {} to {}.'.format(name, old, new))
    return inventory


def _transact(inventory, type):
    if type == 'buy':
        msg1 = 'What item do you want to buy?\n'
        msg2 = 'I don\'t sell that. Choose another item.\n'
        msg3 = 'You can\'t buy zero. How many?\n'
        f = buy_item
    elif type == 'sell':
        msg1 = 'What item do you want to sell?\n'
        msg2 = 'I don\'t buy that. Choose another item.\n'
        msg3 = 'You can\'t sell zero. How many?\n'
        f = sell_item
    item = input(msg1)
    while item not in inventory:
        item = input(msg2)
    quantity = int(input("How many?\n"))
    while quantity <= 0:
        item = input(msg3)
    return f(item, quantity, inventory)


def take_action(inventory):
    action = input(
        "What do you want to do?\n"
        "(Type 'help' for a list of available actions.)\n"
    ).lower()
    if action in actions:
        if action == 'buy':
            inventory = _transact(inventory, 'buy')
        elif action == 'sell':
            inventory = _transact(inventory, 'sell')
        elif action == 'plant':
            inventory = plant_seed(inventory)
        elif action == 'water':
            inventory = water_plant(inventory)
        elif action == 'harvest':
            pass
        elif action == 'inventory':
            check_inventory(inventory)
        elif action == 'help':
            help()
        elif action == 'nothing':
            pass
    else:
        print("You cannot do that.")
    return inventory, action


def get_plant_index(inventory, plantName):
    for i, plant in enumerate(inventory['plants']):
        if plant.name == plantName:
            return i


def things_happen(inventory):
    for i, plant in enumerate(inventory['plants']):
        inventory['plants'][i].dry()
        inventory['plants'][i].photosynthesis()
        inventory['plants'][i].increase_maturity()
        delta = plant.moisture - plant.optimalMoisture

        if not (abs(delta) < plant.moistureRange):
            pr = 'over' if delta > plant.moistureRange else 'under'
            print('{} is {} moistured!'.format(plant.name, pr))
            plant.wither()
            if inventory['plants'][i].dead:
                inventory['plants'].remove(plant)
    return inventory


def help():
    print('You may do any of the following:')
    for action in actions:
        print(action)


def do_nothing():
    pass


def main(inventory):
    action = ''
    while (True):
        if (action != 'inventory' and action != 'help'):
            inventory = things_happen(inventory)
        inventory, action = take_action(inventory)


subdivisionDays = 4
flowersPerPlant = 1
plants = 0
plantsPerBed = 5
plantsPerSprinkler = 10

inventory = {
    'beds':      1,
    'flowers':   0,
    'money':     100,
    'plants':    [],
    'rake':      0,
    'seeds':     10,
    'shovel':    0,
    'sprinkler': 0,
}

prices = {
    'bed':       10,
    'flowers':   10,
    'plants':    20,
    'rake':      5,
    'seeds':     5,
    'shovel':    25,
    'sprinkler': 40,
}

actions = {
    'buy':       buy_item,
    'inventory': check_inventory,
    'plant':     plant_seed,
    'harvest':   harvest_crop,
    'help':      help,
    'nothing':   do_nothing,
    'sell':      sell_item,
    'water':     water_plant,
}

main(inventory)
