"""
Author: Christian Lozoya, 2017
"""

from Utilities.DataHandler import *
from Utilities.MenuFunctions import *

def state_code_state():
    statesDict = {}
    with open(STATES, 'r') as states:
        with open(STATE_CODES, 'r') as state_codes:
            for state, state_code in zip(states, state_codes):
                statesDict[state_code.strip()] = state.strip()
    return statesDict


def set_folders(folders, entries):
    fld = folders
    if entries:
        fld = pd.read_csv(StringIO(entries), sep=',')
    return [str(i)[2:] for i in fld]


def set_files(states, entries):
    files = []
    if entries:
        state_codes = pd.read_csv(StringIO(entries), sep=',')
        for state in states:
            if state[:2] in state_codes or state in state_codes:
                files.append(states[state])
    else:
        files = [states[state] for state in states]
    return files


def search(entries, dtype, file, dir, columns=None):
    data = read_data(DATABASE + os.path.join(dir, file), dtype)
    if not data.empty:
        data = remove_nonmatches(data, entries, columns)
    if not data.empty:
        data = rename_duplicates(data)
        export_results(data, file)
    else:
        return
    return data


def remove_nonmatches(dataFrame, entries, columns=None):
    dF = dataFrame
    if not columns: columns = [entry for entry in entries if entry not in (FILE, FOLDER) and entries[entry].text() is not '']
    if False==True:#if not settings[2].text():  # Union
        pass
    else:  # Intersection
        for column in columns:
            collection = []
            entry = pd.read_csv(StringIO(entries[column].text()), sep=',')
            for i in entry:
                if '<' in i or '>' in i or '-' in i:
                    collection = filter(dF, entry, collection, column)
                else:
                    if not (str(i) in dF[column].values):
                        print("not here")

                    else:
                        print("found")
                        if type(i) == str: i = [i] # homogenizing the type of i
                        collection.append(pd.DataFrame(dF[dF[column].isin(i)]))

            dF = pd.DataFrame(pd.concat([c for c in collection], join='inner', axis=0))
        return dF


def filter(dataFrame, entry, collection, column):
    filtered = collection
    for k in (inequality(entry), interval(entry)):
        for r in k:
            operator = r[0][0]
            if len(r[0]) == 1:  # Inequalities
                temp_df = pd.DataFrame(data=dataFrame[operator(dataFrame[column].astype(float), float(r[1][0]))])
                filtered.append(temp_df)
            elif len(r[0]) == 2:  # Ranges
                operator2 = r[0][1]
                temp_df = pd.DataFrame(data=dataFrame[
                    operator(dataFrame[column].astype(float), float(pd.DataFrame(r[1]).columns[0])) &
                    operator2(dataFrame[column].astype(float), float(pd.DataFrame(r[1]).columns[1]))])
                filtered.append(temp_df)
    return filtered


def inequality(entry):
    for value in entry:
        for operator in OPERATORS:
            if operator in value:
                r = re.findall(r'(?<=[' + operator + '])(\d*[.]?\d*$)', value)
                yield ((OPERATORS[operator],), r)
                break


def interval(entry):
    for value in entry:
        for i in range(INDICATORS.__len__()):
            if INDICATORS[i] in value:
                r = pd.read_csv(StringIO(value), sep=INDICATORS[i], engine='python')
                if INDICATORS[i] == '--':
                    yield ((OPERATORS['>='], OPERATORS['<=']), r)
                elif INDICATORS[i] == '-':
                    yield ((OPERATORS['>'], OPERATORS['<']), r)
                break
