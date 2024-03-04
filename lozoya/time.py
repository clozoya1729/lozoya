import datetime
import time

import timeit

timeMS = lambda: int(round(time.time() * 1000))


def get_today():
    return datetime.datetime.now().date()


def get_next_week():
    return get_today() + datetime.timedelta(days=7)


def format_date(date):
    return date.strftime('%m/%d/%Y')


def ftoday():
    return format_date(get_today())


def fnext_week():
    return format_date(get_next_week())


class Timer:
    def __init__(self):
        pass

    def start(self):
        self.start = timeit.default_timer()

    def stop(self):
        self.stop = timeit.default_timer()

    def display(self):
        print(self.stop - self.start)


def timer(function):
    """
    Outputs the time a function takes
    to execute.
    """

    def wrapper(*args, **kwargs):
        fruit = None
        t1 = time.time()
        try:
            fruit = function(*args, **kwargs)
        except:
            function(*args, **kwargs)
        t2 = time.time()
        print("Time it took to run the function: " + str((t2 - t1)) + "\n")
        if fruit is not None:
            return fruit

    return wrapper
