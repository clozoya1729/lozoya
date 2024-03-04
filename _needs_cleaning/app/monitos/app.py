from random import choice as choice

locations = [
    'Pearl Apartments',
    'Sunrise Apartments',
    'Lincoln Park',
    'Kremlin Beach',
]


class Person():
    def __init__(self, name, gender, birthday, father, mother):
        self.name = name
        self.gender = gender
        self.father = father
        self.mother = mother
        self.age = 0
        self.birthday = birthday
        self.children = 0
        self.alive = True
        self.health = rand(0, 50) + rand(0, 40) + rand(0, 30) + rand(0, 20)
        if self.health > 100:
            self.health = 100
        # print(self.health)
        # relationships: dict
        # key: person, value: percentage
        # 100% ~ Unconditional love
        # 0% ~ Indifference/Unaware
        # -100% ~ Hatred
        self.relationships = {}

        # emotions: dict
        # key: emotion, value: percentage
        self.emotions = {
            'happiness':  50,
            'motivation': 50,
            'sadness':    0,
            'anger':      0
        }

        self.characteristics = {
            'beauty':       50,
            'strength':     50,
            'intelligence': 50,
            'charisma':     50
        }
        self.location = 'Home'
        print(
            self.gender + ' ' + self.name + '  born to ' + self.father + ' and ' + self.mother
        )  # + ', the mother, on ' + str(self.birthday))

    def __repr__(self):
        representation = 'Person ' + self.name
        return representation

    def set_health(self):
        low = -round(self.age * 1.75, 0)
        high = int(500 / (self.age + 1))
        r = rand(low, high) / 10e2

        percentage = self.health + r
        if rand(0, 100) > 90:
            self.health = percentage
        if self.health > 100:
            self.health = 100
        if self.health < 50:
            r = rand(0, 100)
            if r > 90:
                self.health -= 5 * self.age / 10e5
        if self.health <= 0:
            self.alive = False
            print(self.name + ' died. ' + str(self.age) + '.')

    def set_relationship(self, person, percentage):
        self.relationships[person] = percentage

    def set_emotion(self, emotion, percentage):
        self.emotions[emotion] = percentage

    def set_age(self, day):
        if day == self.birthday:
            self.age += 1

    # print(self.name + ' is ' + str(self.age) + ' years old.')

    def set_location(self, locations):
        r = rand(0, 100)
        if r > rand(0, r * r):
            self.location = choice(locations)


# print(self.name + ' is at ' + self.location)

def set_illness(person):
    if person.health:
        pass


def make_baby(parent1, parent2, birthday):
    if parent1.gender == 'male':
        father = parent1.name
        mother = parent2.name
    else:
        father = parent2.name
        mother = parent1.name

    name = ''
    for i in range(rand(4, 11)):
        if i == 0:
            name += (choice('ABCDEFGHIJKLMNOPQRSTVWXYZ'))
        else:
            name += (choice('aaaeeeiiiooouuuabcdefghijklmnopqrstuvwxyz'))
    gender = choice(['male', 'female'])

    return Person(name, gender, birthday, father, mother)


from random import randint as rand

from app.monitos.app import make_baby
from app import Person

amy = Person('Amy', 'female', 25, father='Dad', mother='Mom')
bill = Person(
    'Bill', 'male', 126,
    father='Father', mother='Mother'
)


def simulate():
    people = [amy, bill]
    day = 1
    for i in range(int(1000e100)):
        if i % 10e2 == 0:

            if i % 10e2 == 0:
                # print('Day ' + str(day))
                for person in people:
                    person.set_health()
                    for other in people:
                        if other != person and other.age > 14 and other.age < 60 and person.age > 14 and person.age < 60 and person.children < 4 and other.children < 4 and other.location == person.location:
                            if other.gender != person.gender:
                                if rand(0, 1000) > 990 and len(people) <= 17:
                                    people.append(make_baby(person, other, day))
                                    person.children += 1
                                    other.children += 1
                        if other.location == person.location:
                            if other != person and rand(0, 1000000) > 999999:
                                print(person.name + ' killed ' + other.name)
                                other.alive = False
                    person.set_age(day)  # person.set_location(locations)
                    # print(person.health)
                    if not (person.alive):
                        people.remove(person)

            day += 1
            if day > 365:
                day = 1


simulate()
