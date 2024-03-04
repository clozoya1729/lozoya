def append_list_as_str(field, val):
    if field == '':
        return val
    return ',{}'.format(val)


def match_substring(string, substringList):
    for substring in substringList:
        if substring in string:
            return True
    return False


def str_to_list(stringArg):
    """
    Convert a string of comma-separated values into a list
    stringo: String of comma-separated values
    return: list of str - [str0, ..., strN]
    """
    if stringArg == '':
        return []
    s = stringArg.split(',')
    if s == None:
        return [stringArg]
    return s


def text_formatter(text, width=100):
    """
    Iterates through each character in text
    and store the number of characters that have been
    iterated over so far in a counter variable.
    If the number of characters exceeds the width,
    a new line character is inserted and the counter resets.
    text: str
    width: int
    return: str
    """
    line = 1
    p = ''
    for e, char in enumerate(text):
        if e != len(text) + 1:
            p += (text[e])
            if (e > line * width and text[e] == ' '):
                line += 1
                p += '\n'
    return p
