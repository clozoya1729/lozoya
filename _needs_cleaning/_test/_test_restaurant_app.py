import random

import matplotlib.pyplot as plt
import numpy as np

import app.restaurant.app

numberOfSimulations = 1000
initialMoney = 1000
daysToSimulate = 1825
moneyMatrix = np.zeros((numberOfSimulations, daysToSimulate))
spendMatrix = np.zeros((numberOfSimulations, daysToSimulate))
strawMatrix = np.zeros((numberOfSimulations, daysToSimulate))

if __name__ == '__main__':
    for i in range(numberOfSimulations):
        restaurantObject = app.restaurant.inventory.Restaurant(initialMoney=initialMoney)
        day = 0
        workdays = [0, 1, 2, 3, 4, 5, 6]
        while day < daysToSimulate:
            weekday = day % 7
            if weekday in workdays:
                sales = random.randint(5, 15)
                restaurantObject.routine_work(day, sales)
            else:
                restaurantObject.routine_nonwork(day)
            # forkSeries.append(restaurantObject.inventory['fork'].quantity)
            moneyMatrix[i, day] = restaurantObject.money
            spendMatrix[i, day] = restaurantObject.spend
            strawMatrix[i, day] = restaurantObject.inventory['straw'].quantity
            day += 1
            # time.sleep(1)
    x = range(0, daysToSimulate)
    plt.plot(
        x,
        moneyMatrix.T,
        lw=0.1,
    )
    # plt.plot(
    #     x,
    #     np.min(moneyMatrix.T, 1),
    # )
    # plt.plot(
    #     x,
    #     np.mean(moneyMatrix.T, 1),
    # )
    # plt.plot(
    #     x,
    #     np.max(moneyMatrix.T, 1),
    # )
    #
    # plt.plot(
    #     x,
    #     spendMatrix.T,
    #     lw=0.01,
    # )
    # plt.plot(
    #     x,
    #     np.min(spendMatrix.T, 1),
    #     lw=2,
    # )
    # plt.plot(
    #     x,
    #     np.mean(spendMatrix.T, 1),
    #     lw=2,
    # )
    # plt.plot(
    #     x,
    #     np.max(spendMatrix.T, 1),
    #     lw=2,
    # )

    # plt.plot(
    #     x,
    #     np.mean(strawMatrix.T, 1),
    #     lw=2,
    # )
    plt.legend()
    plt.show()
