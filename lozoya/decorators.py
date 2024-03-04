def catch_error(error_callback: callable, errorMessage: str):
    def inner(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_callback(e, errorMessage, *args, **kwargs)

        return wrapper

    return inner
