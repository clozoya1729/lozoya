from Utilities.Constants import *


def read_folder(dir, ext):
    """
    Collect all file paths with given extension (ext) inside a folder directory (dir)
    """
    paths = []
    for file in os.listdir(dir):
        if file[-4:].lower() == ext.lower():
            paths.append(file)
    return paths


def read_data(path, dtype='str'):
    """
    :param path:
    :param dtype:
    :return:
    """
    with open(path, newline='') as file: header = next(csv.reader(file))
    data = pd.read_csv(path, usecols=[item for item in header if item != FOLDER], na_values=NA_VALUES,
                       dtype=dtype, encoding=ENCODING)
    data.set_index(ID, inplace=True)
    return data


def rename_duplicates(dataFrame):
    """
    Rename duplicates in pandas data frame
    """
    data = dataFrame
    i = 2
    while True in data.index.duplicated():
        data.index = data.index.where(~data.index.duplicated(), data.index + '_' + str(i))
        i += 1
    return data


def clean_data(dataFrame, columns):
    """
    Clears incomplete data resulting in regular matrix
    data is a pandas DataFrame and columns are the columns in data
    """
    cln = dataFrame
    for column in columns:
        cln = (cln[np.isfinite(cln[column].astype(float))])
    cln.columns = columns
    return cln


def concat_data(dataFrame, columns, index=None):
    """
    Concatenate list of pandas Series into a DataFrame
    dataFrame is a pandas DataFrame and columns are the columns in data
    index is used for concatenating only specific columns
    """
    if index:
        cnc = pd.DataFrame(pd.concat([d[index] for d in dataFrame], axis=1))
    else:
        cnc = pd.DataFrame(pd.concat([d for d in dataFrame], axis=1))
    cnc.columns = columns
    return cnc


def normalize_rows(dataFrame):
    """
    Normalize dataFrame by row
    """
    try:
        nrm = dataFrame.div(dataFrame.sum(axis=1), axis=0)
        nrm.fillna(0, inplace=True)
        return nrm
    except:
        print("Division by zero")


def export_results(dataFrame, fileName, csv=True, excel=False, pdf=False):
    if csv:
        dataFrame.to_csv(path_or_buf=os.path.join(fileName) + ' search results' + TXT_EXT)

    if excel:
        try:  # excel workbook exists
            writer = pd.ExcelWriter(os.path.join(fileName) + XLSX_EXT, engine='openpyxl')
            writer.book = load_workbook(os.path.join(fileName) + XLSX_EXT)
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()
        except:  # excel workbook is created if it does not exist
            writer = pd.ExcelWriter(os.path.join(fileName) + XLSX_EXT, engine='xlsxwriter')
            dataFrame.to_excel(writer, sheet_name=fileName)
            writer.save()

    if pdf:
        pass
