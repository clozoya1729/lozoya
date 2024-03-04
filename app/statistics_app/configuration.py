"""Automatic Descriptive Statistics Report Generator"""
# TODO https://stackoverflow.com/questions/23359083/how-to-convert-webpage-into-pdf-by-using-python
import datetime

beginDoc = str('<!DOCTYPE html>\n<html>\n<link rel="stylesheet" href="ReportCSS4.css">\n<body>')
endDoc = str('</body>\n</html>')
indent = "&emsp;&emsp;&emsp;&emsp;"
subSpaceP = "</br></br></br>"


def sub_space(num):
    if num == 0:
        return ""
    s = "</br>"
    for i in range(num):
        s += "</br>"
    return s


def headerize(headerString, headerLevel):
    return str("<h{0}>{1}</h{0}>").format(headerLevel, headerString)


def paragraphize(paragraphString, ind=True, sub=True):
    indentation = ''
    if ind:
        indentation = indent
    if sub == True:
        subS = subSpaceP
    elif type(sub) == type(1):
        subS = sub_space(sub)
    else:
        subS = ''

    return str("<p>{0}" + paragraphString + "{1}</p>").format(indentation, subS)


def generate_report(title, client, analyst, mean, median, mode, dataRange, std, variance, skew, kurtosis):
    """
    title: str, title of the report
    client: str, client requesting analysis and report
    analyst: str, analyst performing analysis and preparing report
    mean: list containing the mean of each variable in the data
    median: list containing the median of each variable in the data
    mode: list containing the mode of each variable in the data
    std: list containing the standard deviation of each variable in the data
    variance: list containing the variance of each variable in the data
    skew: list containing the skew of each variable in the data
    kurtosis: list containing the kurtosis of each variable in the data
    """
    distTOC = "1"
    ctTOC = "1"
    dispTOC = "2"
    corrTOC = "3"
    headers = {
        "DISTRIBUTION":       distTOC,
        "CENTRAL TENDENCIES": ctTOC,
        "Mean":               ctTOC + ".1",
        "Median":             ctTOC + ".1",
        "Mode":               ctTOC + ".2",
        "DISPERSION":         dispTOC,
        "Range":              dispTOC + ".1",
        "Standard Deviation": dispTOC + ".1",
        "Variance":           dispTOC + ".2",
        "Skew":               dispTOC + ".3",
        "Kurtosis":           dispTOC + ".5",
        "CORRELATION":        corrTOC
    }
    """Basic descriptions (section introductions)"""
    intros = {
        "distribution": paragraphize(
            str(
                'The distribution of a variable is all of the possible values that variable can be as well as the likelihood of it being that value.'
            )
        ),
        "ct":           paragraphize(
            str('The Central Tendencies of a distribution are described by the mean, median, and mode.')
        ),
        "mean":         paragraphize(str('The mean of a distribution describes...')),
        "median":       paragraphize(
            str(
                'The median of a distribution is the value that divides the data exactly in half. Therefore, the median is the numerical value for which 50% of the data is above and 50% is below.'
            )
        ),
        "mode":         paragraphize(
            str(
                'The mode of a distribution is its peak value. This represents the value which is more likely than any search value.'
            )
        ),
        "dispersion":   paragraphize(
            str(
                'The Dispersion of a distribution is described by the standard deviation or variance, skew, and kurtosis.'
            )
        ),
        "range":        paragraphize(str('The range of a distribution is the minimum and maximum values.')),
        "std":          paragraphize(str('The standard deviation of a distribution describes...')),
        "variance":     paragraphize(str('The variance of a distribution describes...')),
        "skew":         paragraphize(
            str(
                'The skew of a distribution describes... A skew of 0 represents a symmetric distribution. A skew higher than 0 means the distribution is... A skew below 0 means the distribution is...'
            )
        ),
        "kurtosis":     paragraphize(
            str(
                'The kurtosis of a distribution describes... A univariate Normal (Gaussian) distribution has a kurtosis of 2, so this is used as a baseline reference. A kurtosis of 2 represents a symmetric distribution. A kurtosis higher than 2 means the distribution is... A kurtosis below 2 means the distribution is...'
            )
        ),
        "correlation":  paragraphize(str('Correlation between two variables describes...'))
    }

    def get_date_text(date):
        dateNum = date.strftime('%d/%m/%Y')
        dateStr = date.strftime('%b %d, %Y')
        return dateNum, dateStr

    def get_title_text():
        """Makes HTML2 file. Populates Title information. Formats and exports as PDF"""
        titlePage = "</br></br></br></br></br>" + \
                    headerize(title, 1) + "</br></br>" + \
                    headerize("Submitted to:", 2) + \
                    headerize(client, 3) + "</br></br>" + \
                    headerize("Prepared by:", 2) + \
                    headerize(analyst, 3) + "</br></br>" + \
                    headerize("Date:", 2) + \
                    headerize(dateStr, 3) + "</br></br></br>"
        return titlePage

    def get_dist_text():
        """Makes HTML2 file. Populates Distribution information. Formats and exports as PDF"""
        """explain the distribution of the data and which analytical distributions fit"""
        header = str(headerize("{1} DISTRIBUTION", 2) + "{0}").format(sub_space(4), headers["DISTRIBUTION"])
        desc = header
        return desc

    def get_ct_text():
        """Makes HTML2 file. Populates Central Tendencies information. Formats and exports as PDF"""

        def get_mean_desc(mean):
            """Returns a description of the mean and its implications"""
            desc = str("\n\n" + headerize("{1} Mean", 3) + "{0}").format(sub_space(0), headers["Mean"])
            desc = desc + intros["mean"]
            return desc

        def get_median_desc(median):
            """Returns a description of the median and its implications"""
            desc = str("\n\n" + headerize("{1} Median", 3) + "{0}").format(sub_space(0), headers["Median"])
            desc = desc + intros["median"]
            return desc

        def get_mode_desc(mode):
            """Returns a description of the mode and its implications"""
            desc = str("\n\n" + headerize("{1} Mode", 3) + "{0}").format(sub_space(0), headers["Mode"])
            desc = desc + intros["mode"]
            return desc

        header = str(headerize("{1} CENTRAL TENDENCIES", 2) + "{0}").format(sub_space(4), headers["CENTRAL TENDENCIES"])
        ctDesc = header + intros["ct"] + "\n" + get_mean_desc(mean) + "\n" + \
                 get_median_desc(median) + "\n" + \
                 get_mode_desc(mode) + "\n"
        return ctDesc

    def get_disp_text():
        """Makes HTML2 file. Populates Dispersion information. Formats and exports as PDF"""

        def get_range_desc(dataRange):
            """Returns a description of the range and its implications"""
            desc = str("\n\n" + headerize("{1} Range", 3) + "{0}").format(sub_space(0), headers["Range"])
            desc = desc + intros["range"]
            return desc

        def get_std_desc(std):
            """Returns a description of the std and its implications"""
            desc = str("\n\n" + headerize("{1} Standard Deviation", 3) + "{0}").format(
                sub_space(0),
                headers["Standard Deviation"]
            )
            desc = desc + intros["std"]
            return desc

        def get_variance_desc(variance):
            """Returns a description of the variance and its implications"""
            desc = str("\n\n" + headerize("{1} Variance", 3) + "{0}").format(sub_space(0), headers["Variance"])
            desc = desc + intros["variance"]
            return desc

        def get_skew_desc(skew):
            desc = str("\n\n" + headerize("{1} Skew", 3) + "{0}").format(sub_space(0), headers["Skew"])
            desc = desc + intros["skew"]
            """if skew > 0:
              desc = desc+str("The skew is {0}, which is greater than 0, so the distribution is skewed to the right."+"\n"+\
              "This means that the tail of the distribution"+"\n"+\
              "extends to the right, and the mass of the distribution"+"\n"+\
              "is concentrated to toward the left.").format(skew)
            elif skew < 0:
              desc = desc+str("The skew is {0}, which is less than 0, so the distribution is skewed to the left."+"\n"+\
              "This means that the tail of the distribution"+"\n"+\
              "extends to the right, and the mass of the distribution"+"\n"+\
              "is concentrated to toward the right.").format(skew)
            else:
              desc = desc+str("The skew is {0}, and there is no skew in the distribution."+"\n"+\
              "This means that the tail of the distribution"+"\n"+\
              "extends equally to both sides and the mass of the distribution"+"\n"+\
              "is centered. This further means that the distribution is symmetric and its mean and median are equivalent.").format(skew)
            """
            return desc

        def get_kurtosis_desc(kurtosis):
            """Returns a description of the kurtosis and its implications"""
            desc = str("\n\n" + headerize("{1} Kurtosis", 3) + "{0}").format(sub_space(0), headers["Kurtosis"])
            desc = desc + intros["kurtosis"]
            """
            if kurtosis> 2:
              desc = desc+str("The kurtosis is {0}, which is greater than 0, so the distribution is skewed to the right."+"\n"+\
              "This means that the tail of the distribution"+"\n"+\
              "extends to the right, and the mass of the distribution"+"\n"+\
              "is concentrated to toward the left.").format(skew)
            elif kurtosis< 2:
              desc = desc+str("The kurtosis is {0}, which is less than 0, so the distribution is skewed to the left."+"\n"+\
              "This means that the tail of the distribution"+"\n"+\
              "extends to the right, and the mass of the distribution"+"\n"+\
              "is concentrated to toward the right.").format(skew)
            else:
              desc = desc+str("The kurtosis is {0}, and there is no skew in the distribution."+"\n"+\
              "This means that the tail of the distribution"+"\n"+\
              "extends equally to both sides and the mass of the distribution"+"\n"+\
              "is centered. This further means that the distribution is symmetric and its mean and median are equivalent.").format(skew)
            """
            return desc

        desc = str(headerize("{1} DISPERSION", 2) + "{0}").format(sub_space(4), headers["DISPERSION"])
        dispersionDesc = desc + intros["dispersion"] + "\n" + \
                         get_range_desc(dataRange) + "\n" + \
                         get_std_desc(std) + "\n" + \
                         get_variance_desc(variance) + "\n" + \
                         get_skew_desc(skew) + "\n" + \
                         get_kurtosis_desc(kurtosis) + "\n"
        return dispersionDesc

    def get_corr_text():
        """Makes HTML2 file. Populates correlation information. Formats and exports as PDF"""
        """explain the correlation between variables if they are above a certain threshold"""
        header = str(headerize("{1} CORRELATION", 2) + "{0}").format(sub_space(4), headers["CORRELATION"])
        desc = header
        return desc

    dateNum, dateStr = get_date_text(datetime.date.today())
    titleText = get_title_text()
    distText = get_dist_text()
    ctText = get_ct_text()
    dispText = get_disp_text()
    corrText = get_corr_text()
    with open('TitlePage.html', 'w') as f:
        f.write(beginDoc)
        f.write(titleText)
        f.write(endDoc)
    with open('Distribution.html', 'w') as f:
        f.write(beginDoc)
        f.write(distText)
        f.write(endDoc)

    with open('CentralTendencies.html', 'w') as f:
        f.write(beginDoc)
        f.write(ctText)
        f.write(endDoc)

    with open('Dispersion.html', 'w') as f:
        f.write(beginDoc)
        f.write(dispText)
        f.write(endDoc)

    with open('Correlation.html', 'w') as f:
        f.write(beginDoc)
        f.write(corrText)
        f.write(endDoc)


stats = [0, 0, 0, 0, 0, 0, 0, 0]
generate_report('Hydraulic Data Statistical Analysis', 'Dannenbaum LLC', 'LoPar Technologies LLC', *stats)

'''
import pandas as pd
data = pd.read_csv(r'Z:\Family\Transfer\sensor_data.csv')
data.set_index(data.columns[0], inplace=True)
data = DataProcessor.optimize_dataFrame(data)
dtypes = DataProcessor.get_dtypes(data)
write_chapter(data, dtypes)
data = pd.read_csv(r'Z:\Family\Transfer\sensor_data.csv')
data.set_index(data.columns[0], inplace=True)
data = DataProcessor.optimize_dataFrame(data)
dtypes = DataProcessor.get_dtypes(data)
write_chapter(data, dtypes)
data = pd.read_csv(r'Z:\Family\Transfer\sensor_data.csv')
data.set_index(data.columns[0], inplace=True)
data = DataProcessor.optimize_dataFrame(data)
dtypes = DataProcessor.get_dtypes(data)
write_chapter(data, dtypes)
data = pd.read_csv(r'Z:\Family\Transfer\sensor_data.csv')
data.set_index(data.columns[0], inplace=True)
data = DataProcessor.optimize_dataFrame(data)
dtypes = DataProcessor.get_dtypes(data)
write_chapter(data, dtypes)
'''

paragraphs = [5, 1, 1, 4]
variations = 2
title = 'Error Metrics'
abstract = True
# Images
PLOTS_DIR = os.path.join('Z:/Family/LoParTechnologies/PlotGenerator/ExpressionStrings')
mae = os.path.join('"' + PLOTS_DIR, 'Mean Absolute Error.png"')
mbe = os.path.join('"' + PLOTS_DIR, 'Mean Bias Error.png"')
mse = os.path.join('"' + PLOTS_DIR, 'Mean Squared Error.png"')
rmse = os.path.join('"' + PLOTS_DIR, 'Root Mean Squared Error.png"')
mae = os.path.join('"FunctionStrings', 'Mean Absolute Error.png"')
mbe = os.path.join('"FunctionStrings', 'Mean Bias Error.png"')
mse = os.path.join('"FunctionStrings', 'Mean Squared Error.png"')
rmse = os.path.join('"FunctionStrings', 'Root Mean Squared Error.png"')
q2 = os.path.join('"FunctionStrings', 'Second Quartile.png"')
q3 = os.path.join('"FunctionStrings', 'Third Quartile.png"')
iqr = os.path.join('"FunctionStrings', 'Interquartile Range.png"')
lowB = os.path.join('"FunctionStrings', 'Lower Bound.png"')
uppB = os.path.join('"FunctionStrings', 'Upper Bound.png"')

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Absolute Error (MAE)
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
MAE of a data set is the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

s112 = h + '''
<p>
MAE of a data set measures the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

# Sentence 1
s121 = '''
<p>
It is the average magnitude 
of the errors in a set of predictions. Therefore, direction is not considered. 
</p>
'''

s122 = '''
<p>
Without their direction being considered, MAE averages the magnitude 
of the errors in a set of predictions.
</p>
'''

# Sentence 2
s131 = '''
<p>
####
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
####
</p>
'''

s132 = '''
####
<p>
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
</p>
'''

# Sentence 3
s141 = '''
<p>
####
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''

s142 = '''
<p>
###
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''
# Sentence 5
s151 = '''
<p>
MAE is calculated as follows:
</p>
''' + img

s152 = '''
'<p>
MAE is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Bias Error (MBE)
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

s211 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

s212 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

# PARAGRAPH 2
# Sentence 1
h = '''
<h3>
Mean Squared Error (MSE)
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''
<p>
The mean squared error is calculated as follows:
</p>
''' + img

s312 = h + '''
<p>
The mean squared error is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 3
# Sentence 1
h = '''
<h3>
Root Mean Squared Error (RMSE)
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

s412 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

# Sentence 1
s421 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

s422 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

# Sentence 2
s431 = '''
<p>
RMSE can be appropriate when large errors should be penalized more than smaller errors.
The penalty of the error changes in a non-linear way.
</p>
'''

s432 = '''
<p>
###
If being off by 10 is more than twice as bad as being off by 5, RMSE can be more appropriate 
in some cases penalizing large errors more.
If being off by 10 is just twice as bad as being off by 5, MAE is more appropriate.
</p>
####
'''

# Sentence 3
s441 = '''
<p>
The root mean square error is calculated as follows:
</p>
''' + img

s442 = '''
<p>
The root mean square error is calculated using the following formula:
</p>
''' + img

"""

import os
import sys
from collections import collections.OrderedDict
import ReporGenerator
import random

def error_metrics_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'Error metrics compare the Regression6 analysis to the data and are used to quantify the uncertainty of "errors" in the Regression6 analysis',
        'Error metrics are used to quantify the uncertainty of "error" in the Regression6 analysis by comparing the Regression6 analysis with respect to the data'
    ])
     p0['sentence 1'] = random.choice([
         'The following are all of the error metrics included in this report',
         'This report includes all of the following error metrics'
     ])
    return [p0]

def mean_absolute_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'MAE of a data set is the average distance between each data value and the mean of the data set',
        'MAE of a data set measures the average distance between each data value and the mean of a data set',
    ])
    p0['sentence 1'] = random.choice([
        'It describes the variation and dispersion in a data set',
        'It describes the variation and dispersion in a data set ',
    ])
    p0['sentence 1'] = random.choice([
        'It is the average magnitude of the errors in a a set of predictions. Therefore, direction is not considered',
        'Without their direction being considered, MAE averages the magnitude of the errors in a set of predictions',
    ])
    p0['sentence 2'] = random.choice([
        'Its the average over the test sample of the absolute differences between prediction and actual observation shere all indivisual differences have equal weight.',
        'Its the average over the test sample of the absolute differencces between prediction and actual observation where all individual differences have equal weight.',
    ])

    p1 = ReporGenerator.Paragraph('paragraph 1')
    p1['sentence 0'] = random.choice([
        'MAE is more appropriate by being off by 00 is just twice as bad as being off by 3, then',
    ])
    p1['sentence 1'] = random.choice([
        'MAE is calculated as follows:',
        'MAE is calculated using the following fomrula:',
    ])
    return [p0,
            p1]

def mean_bias_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'If the absolute value is not taken (the signs of the errors are not removed), the average error becomes the Mean Bias Error (MBE) and is usually intended to measure model bias.'
        'If the absolute value is not taken (the signs of the errors are not removed), the average error becomes the Mean Bias Error (MBE) and is usuallly intended to measure average model bias'
    ])
    p0['sentence 1'] = random.choice([
        'MBE can convey useful information, but should be interpreted cautiously because positive and negative errors will cancel out',
        'MBE can convey useful information, but should be interpreted cautiously because positive and negative errors will cancel out.'
    ])

    return [p0]



def mean_squared_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'The mean squared error is calculated as follows:'
        'The mean squared error is calculated using the following formula:'
    ])

    return[p0]


def root_mean_squared_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'RMSE is a quadratic scoring rule that also measures the average magnitude of the error.',
        'RMSE is a quadratic scoring rule that also measures the average magnitude of the error'
          ])
    p0['sentence 1'] = random.choice([
        'Its the square root of the average of squared differences between prediciton and actual observation.',
        'Its the square root of the average of squared differences between prediction and actual observation'
    ])
    p0['sentence 1'] = random.choice([
        'RMSE does not necessarily increase with the variance of the errors'
    ])
    p0['sentence 2'] = random.choice([
        'RMSE is proportional to the variance of the error magnitude frequancy distribution.'
    ])
    p0['sentence 3'] = random.choice([
        'However, it is not proportional to the error variance.'
    ])
    p1 = ReportGenerator.Paragraph('paragraph 1')
    p1['sentence 0'] = random.choice([
        'RMSE can be appropriate when large errors should be penalized more than smaller errors. The penalty of the rror changes in a non-linear way.'
    ])
    p1['sentence 1'] = random.choice([
        'If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate in some cases penalizing large errors more.',
        'If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate'
    ])
    p1['sentence 1'] = random.choice([
        'The root mean square is calculated as follows:'
        'The root mean square is calculated using the following formula:'
    ])
    return[p0,
           p1]

def create_section():
    section = ReportGenerator.Section(title)
    description = ReportGenerator.Subsection('Descritpion')
    description.insert_content(globals()['()_description'.format(title.lower())]())
    test = ReportGenerator.Subsection('Example')
    section.insert_subsections([description,
                                test])
    return section

def write_chapter():

    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Subsections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections can contain of Figure, Caption, or paragraph)

    chapter = ReportGenerator.Chapter('Descriptive Statistics')
    error_metrics = create_section('error metrics')
    mean_absolute_error = create_section('mean absolute error')
    mean_bias_error = create_section('mean bias error')
    mean_squared_error = create_section('mean squared error')
    root_mean_squared error = create_section('root mean squared error')

"""

write_chapter()

"""

import os
import sys
from collections import collections.OrderedDict
import ReporGenerator
import random

def error_metrics_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'Error metrics compare the Regression6 analysis to the data and are used to quantify the uncertainty of "errors" in the Regression6 analysis',
        'Error metrics are used to quantify the uncertainty of "error" in the Regression6 analysis by comparing the Regression6 analysis with respect to the data'
    ])
     p0['sentence 1'] = random.choice([
         'The following are all of the error metrics included in this report',
         'This report includes all of the following error metrics'
     ])
    return [p0]

def mean_absolute_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'MAE of a data set is the average distance between each data value and the mean of the data set',
        'MAE of a data set measures the average distance between each data value and the mean of a data set',
    ])
    p0['sentence 1'] = random.choice([
        'It describes the variation and dispersion in a data set',
        'It describes the variation and dispersion in a data set ',
    ])
    p0['sentence 1'] = random.choice([
        'It is the average magnitude of the errors in a a set of predictions. Therefore, direction is not considered',
        'Without their direction being considered, MAE averages the magnitude of the errors in a set of predictions',
    ])
    p0['sentence 2'] = random.choice([
        'Its the average over the test sample of the absolute differences between prediction and actual observation shere all indivisual differences have equal weight.',
        'Its the average over the test sample of the absolute differencces between prediction and actual observation where all individual differences have equal weight.',
    ])

    p1 = ReporGenerator.Paragraph('paragraph 1')
    p1['sentence 0'] = random.choice([
        'MAE is more appropriate by being off by 00 is just twice as bad as being off by 3, then',
    ])
    p1['sentence 1'] = random.choice([
        'MAE is calculated as follows:',
        'MAE is calculated using the following fomrula:',
    ])
    return [p0,
            p1]

def mean_bias_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'If the absolute value is not taken (the signs of the errors are not removed), the average error becomes the Mean Bias Error (MBE) and is usually intended to measure model bias.'
        'If the absolute value is not taken (the signs of the errors are not removed), the average error becomes the Mean Bias Error (MBE) and is usuallly intended to measure average model bias'
    ])
    p0['sentence 1'] = random.choice([
        'MBE can convey useful information, but should be interpreted cautiously because positive and negative errors will cancel out',
        'MBE can convey useful information, but should be interpreted cautiously because positive and negative errors will cancel out.'
    ])

    return [p0]



def mean_squared_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'The mean squared error is calculated as follows:'
        'The mean squared error is calculated using the following formula:'
    ])

    return[p0]


def root_mean_squared_error_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice([
        'RMSE is a quadratic scoring rule that also measures the average magnitude of the error.',
        'RMSE is a quadratic scoring rule that also measures the average magnitude of the error'
          ])
    p0['sentence 1'] = random.choice([
        'Its the square root of the average of squared differences between prediciton and actual observation.',
        'Its the square root of the average of squared differences between prediction and actual observation'
    ])
    p0['sentence 1'] = random.choice([
        'RMSE does not necessarily increase with the variance of the errors'
    ])
    p0['sentence 2'] = random.choice([
        'RMSE is proportional to the variance of the error magnitude frequancy distribution.'
    ])
    p0['sentence 3'] = random.choice([
        'However, it is not proportional to the error variance.'
    ])
    p1 = ReportGenerator.Paragraph('paragraph 1')
    p1['sentence 0'] = random.choice([
        'RMSE can be appropriate when large errors should be penalized more than smaller errors. The penalty of the rror changes in a non-linear way.'
    ])
    p1['sentence 1'] = random.choice([
        'If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate in some cases penalizing large errors more.',
        'If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate'
    ])
    p1['sentence 1'] = random.choice([
        'The root mean square is calculated as follows:'
        'The root mean square is calculated using the following formula:'
    ])
    return[p0,
           p1]

def create_section():
    section = ReportGenerator.Section(title)
    description = ReportGenerator.Subsection('Descritpion')
    description.insert_content(globals()['()_description'.format(title.lower())]())
    test = ReportGenerator.Subsection('Example')
    section.insert_subsections([description,
                                test])
    return section

def write_chapter():

    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Subsections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections can contain of Figure, Caption, or paragraph)

    chapter = ReportGenerator.Chapter('Descriptive Statistics')
    error_metrics = create_section('error metrics')
    mean_absolute_error = create_section('mean absolute error')
    mean_bias_error = create_section('mean bias error')
    mean_squared_error = create_section('mean squared error')
    root_mean_squared error = create_section('root mean squared error')

"""

# from Utilities14 import headerize, paragraphize

'''
EXAMPLE

paragraphs = 1
sentencesInParagraphs = [2, 3]
variations = 3



# PARAGRAPH 1
# Sentence 1
s111 = 'Once, there was a small tiger.'
s112 = 'There was once a small tiger.'
s113 =  'A small tiger once was.'
s114 = 'This story is about a diminutive tiger.'

# Sentence 1
s121 = 'Unlike most tigers, this tiger was incredibly small!'
s122 = 'The size of this tiger was unimaginably tiny!'
s123 = 'Indeed, this one was not large at all!'
s124 = 'It was remarkably small for being a tiger.'

# Sentence 2
s131 = 'The tiger graduated college after much hard work.'
s132 = 'After much hard work, the tiger graduated college.'
s133 = 'The tiger worked very hard and graduated college.'
s134 = 'The tiger was able to graduate college through a lot of hard work.'


# PARAGRAPH 1
# Sentence 1
s211 = 'The tiger\'s mother was too happy.'
s212 = 'This made the tiger\'s mother so happy.'
s213 = 'Its mother couldn\'t be happier.'
s214 = 'Saying that the tiger\'s mother was happy upon hearing this is an understatement.'

# Sentence 1
s221 = 'Mother tiger bought the tiny tiger a truck.'
s222 = 'The little tiger was given a truck by its mother.'
s223 = 'Mother tiger bought the tiny tiger a truck.'
s224 = 'The little tiger was given a truck by its mother.'

# Sentence 2
s231 = 'The tiger was killed in a violent attack.'
s232 = 'An attack against the tiger lead to its death.'
s233 = 'Unfortunately, this tiger did not live to old age due to a fatal attack.'
s234 = 'An unexpected and violent attack cut its life short.'

# Sentence 3
s241 = 'The story ends here.'
s242 = 'This is the end of the story.'
s243 = 'The end.'
s244 = 'This concludes the story.'
'''

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Absolute Error (MAE)
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
MAE of a data set is the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

s112 = h + '''
<p>
MAE of a data set measures the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

# Sentence 1
s121 = '''
<p>
It is the average magnitude 
of the errors in a set of predictions. Therefore, direction is not considered. 
</p>
'''

s122 = '''
<p>
Without their direction being considered, MAE averages the magnitude 
of the errors in a set of predictions.
</p>
'''

# Sentence 2
s131 = '''
<p>
####
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
####
</p>
'''

s132 = '''
####
<p>
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
</p>
'''

# Sentence 3
s141 = '''
<p>
####
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''

s142 = '''
<p>
###
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''
# Sentence 5
s151 = '''
<p>
MAE is calculated as follows:
</p>
''' + img

s152 = '''
'<p>
MAE is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Bias Error (MBE)
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

s211 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

s212 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

# PARAGRAPH 2
# Sentence 1
h = '''
<h3>
Mean Squared Error (MSE)
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''
<p>
The mean squared error is calculated as follows:
</p>
''' + img

s312 = h + '''
<p>
The mean squared error is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 3
# Sentence 1
h = '''
<h3>
Root Mean Squared Error (RMSE)
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

s412 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

# Sentence 1
s421 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

s422 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

# Sentence 2
s431 = '''
<p>
RMSE can be appropriate when large errors should be penalized more than smaller errors.
The penalty of the error changes in a non-linear way.
</p>
'''

s432 = '''
<p>
###
If being off by 10 is more than twice as bad as being off by 5, RMSE can be more appropriate 
in some cases penalizing large errors more.
If being off by 10 is just twice as bad as being off by 5, MAE is more appropriate.
</p>
####
'''

# Sentence 3
s441 = '''
<p>
The root mean square error is calculated as follows:
</p>
''' + img

s442 = '''
<p>
The root mean square error is calculated using the following formula:
</p>
''' + img

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Absolute Error (MAE)
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
MAE of a data set is the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

s112 = h + '''
<p>
MAE of a data set measures the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

# Sentence 1
s121 = '''
<p>
It is the average magnitude 
of the errors in a set of predictions. Therefore, direction is not considered. 
</p>
'''

s122 = '''
<p>
Without their direction being considered, MAE averages the magnitude 
of the errors in a set of predictions.
</p>
'''

# Sentence 2
s131 = '''
<p>
####
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
####
</p>
'''

s132 = '''
####
<p>
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
</p>
'''

# Sentence 3
s141 = '''
<p>
####
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''

s142 = '''
<p>
###
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''
# Sentence 5
s151 = '''
<p>
MAE is calculated as follows:
</p>
''' + img

s152 = '''
'<p>
MAE is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Bias Error (MBE)
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

s211 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

s212 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

# PARAGRAPH 2
# Sentence 1
h = '''
<h3>
Mean Squared Error (MSE)
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''
<p>
The mean squared error is calculated as follows:
</p>
''' + img

s312 = h + '''
<p>
The mean squared error is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 3
# Sentence 1
h = '''
<h3>
Root Mean Squared Error (RMSE)
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

s412 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

# Sentence 1
s421 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

s422 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

# Sentence 2
s431 = '''
<p>
RMSE can be appropriate when large errors should be penalized more than smaller errors.
The penalty of the error changes in a non-linear way.
</p>
'''

s432 = '''
<p>
###
If being off by 10 is more than twice as bad as being off by 5, RMSE can be more appropriate 
in some cases penalizing large errors more.
If being off by 10 is just twice as bad as being off by 5, MAE is more appropriate.
</p>
####
'''

# Sentence 3
s441 = '''
<p>
The root mean square error is calculated as follows:
</p>
''' + img

s442 = '''
<p>
The root mean square error is calculated using the following formula:
</p>
''' + img

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Absolute Error (MAE)
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
MAE of a data set is the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

s112 = h + '''
<p>
MAE of a data set measures the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

# Sentence 1
s121 = '''
<p>
It is the average magnitude 
of the errors in a set of predictions. Therefore, direction is not considered. 
</p>
'''

s122 = '''
<p>
Without their direction being considered, MAE averages the magnitude 
of the errors in a set of predictions.
</p>
'''

# Sentence 2
s131 = '''
<p>
####
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
####
</p>
'''

s132 = '''
####
<p>
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
</p>
'''

# Sentence 3
s141 = '''
<p>
####
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''

s142 = '''
<p>
###
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''
# Sentence 5
s151 = '''
<p>
MAE is calculated as follows:
</p>
''' + img

s152 = '''
'<p>
MAE is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Bias Error (MBE)
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

s211 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

s212 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

# PARAGRAPH 2
# Sentence 1
h = '''
<h3>
Mean Squared Error (MSE)
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''
<p>
The mean squared error is calculated as follows:
</p>
''' + img

s312 = h + '''
<p>
The mean squared error is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 3
# Sentence 1
h = '''
<h3>
Root Mean Squared Error (RMSE)
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

s412 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

# Sentence 1
s421 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

s422 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

# Sentence 2
s431 = '''
<p>
RMSE can be appropriate when large errors should be penalized more than smaller errors.
The penalty of the error changes in a non-linear way.
</p>
'''

s432 = '''
<p>
###
If being off by 10 is more than twice as bad as being off by 5, RMSE can be more appropriate 
in some cases penalizing large errors more.
If being off by 10 is just twice as bad as being off by 5, MAE is more appropriate.
</p>
####
'''

# Sentence 3
s441 = '''
<p>
The root mean square error is calculated as follows:
</p>
''' + img

s442 = '''
<p>
The root mean square error is calculated using the following formula:
</p>
''' + img

paragraphs = [5, 1, 1, 4]
variations = 2
title = 'Error Metrics'
abstract = True

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Absolute Error (MAE)
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
MAE of a data set is the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

s112 = h + '''
<p>
MAE of a data set measures the average distance between 
each data value and the mean of a data set. 
It describes the variation and dispersion in a data set.
</p>
'''

# Sentence 1
s121 = '''
<p>
It is the average magnitude 
of the errors in a set of predictions. Therefore, direction is not considered. 
</p>
'''

s122 = '''
<p>
Without their direction being considered, MAE averages the magnitude 
of the errors in a set of predictions.
</p>
'''

# Sentence 2
s131 = '''
<p>
####
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
####
</p>
'''

s132 = '''
####
<p>
It’s the average over the test sample of the absolute 
differences between prediction and actual observation where 
all individual differences have equal weight.
</p>
'''

# Sentence 3
s141 = '''
<p>
####
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''

s142 = '''
<p>
###
MAE is more appropriate if being off by 10 is just twice as bad as being off by 5, then .
</p>
'''
# Sentence 5
s151 = '''
<p>
MAE is calculated as follows:
</p>
''' + img

s152 = '''
'<p>
MAE is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean Bias Error (MBE)
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

s211 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

s212 = h + '''
<p>
####
If the absolute value is not taken (the signs of the errors are not removed), 
the average error becomes the Mean Bias Error (MBE) and is 
usually intended to measure average model bias. 
MBE can convey useful information, but should be interpreted cautiously 
because positive and negative errors will cancel out.
####
</p>
''' + img

# PARAGRAPH 2
# Sentence 1
h = '''
<h3>
Mean Squared Error (MSE)
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''
<p>
The mean squared error is calculated as follows:
</p>
''' + img

s312 = h + '''
<p>
The mean squared error is calculated using the following formula:
</p>
''' + img

# PARAGRAPH 3
# Sentence 1
h = '''
<h3>
Root Mean Squared Error (RMSE)
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

s412 = h + '''
#### 
<p>
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
It’s the square root of the average of squared differences between prediction and actual observation.
</p>
###
'''

# Sentence 1
s421 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

s422 = '''
<p>
RMSE does not necessarily increase with the variance of the errors. 
RMSE is proportional to the variance of the error magnitude frequency distribution.
However, it is not necessarily proporional to the error variance.
</p>
'''

# Sentence 2
s431 = '''
<p>
RMSE can be appropriate when large errors should be penalized more than smaller errors.
The penalty of the error changes in a non-linear way.
</p>
'''

s432 = '''
<p>
###
If being off by 10 is more than twice as bad as being off by 5, RMSE can be more appropriate 
in some cases penalizing large errors more.
If being off by 10 is just twice as bad as being off by 5, MAE is more appropriate.
</p>
####
'''

# Sentence 3
s441 = '''
<p>
The root mean square error is calculated as follows:
</p>
''' + img

s442 = '''
<p>
The root mean square error is calculated using the following formula:
</p>
''' + img

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
# MEAN
h = '''
<h3>
Mean
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
The Mean of your data set is also known as the average of your data set.
</p>
'''

s112 = h + '''
<p>
The Average of your data set is what is called the Mean or the central value of the data set.
</p>
'''

s113 = h + '''
<p>
The Mean is the average of all the data points within the data set and is sometime called the Arithmetic Mean.
</p>
'''

s114 = h + '''
<p>
One way to find the Central Tendency of your data is to find the Mean (average) of your data set. 
</p>
'''

s115 = h + '''
<p>
The Mean is the Average of your data set
</p>
'''

s116 = h + '''
<p>
The average of all the data points within the data set is called the Mean of the the data set. 
</p>
'''

s117 = h + '''
<p>
The mean is just another term for the average of the data set.
</p>
'''

# Sentence 1
s121 = '''
<p> 
To get the Mean you must add up all the data points. take the sum and divide by the total number of data points. 
</p>
'''

s122 = '''
<p>
The Mean is the summation of the data set points divided by the number of dataset points. 
</p>
'''

s123 = '''
<p>
To calculate the Average (Mean), add together all the data points within the data set, and then divide the sum by the total number of data points within the data set.
</p>
'''

s124 = '''
<p>
The average is calculated by first adding up all the data points within your data set. Take the sum and divide it by the total number of data points within youur set. 
</p>
'''

s125 = '''
<p>
You must first add up the data points within your data set and then divide by the total number of data points. 
</p>
'''

s126 = '''
<p>
The mean is calculated by adding up all the data points and dividing by the total number of data points. 
</p>
'''

s127 = '''
<p>
Add all the values together and divide them by the number of data points within the data set. 
</p>
'''

# Sentence 2
s131 = '''
<p>
By doing this you have a better understanding of what the average of your data set is.
</p>
'''

s132 = '''
####
<p>
The Mean is one way to find the Central Tendency of your data set. The central tendency can be used to define larger data sets by one value. 
</p>
'''

s133 = '''
<p>
Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number. 
</p>
'''

s134 = '''
####
<p>
Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number. 
</p>
'''

s135 = '''
####
<p>
This gives you a central number to define your data
</p>
'''

s136 = '''
####
<p>
This gives you a central number to define your data
</p>
'''

# Paragraph2
# MEDIAN

h = '''
<h3>
Median 
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

# Sentence 1
s211 = h + '''
<p>
The Median is the midpoint of the data set. It is the point that separates the lower half of your data from the top half.
</p>
'''

s212 = h + '''
<p>
The midpoint of the data set is known as the Median. The Median is used to represent the center of your data set.
</p>
'''

s213 = h + '''
<p>
Your data can be split into two sets, a lower half and a top half. The midpoint of the set is known as the Median
\</p>
'''

s214 = h + '''
'<p>
The Median is the data point lying at the midpoint of the data set.
</p>
'''

s215 = h + '''
'<p>
The Median is the center value within a data set.
</p>
'''

s216 = h + '''
'<p>
The median is the middle value of the data set
</p>
'''

s217 = h + '''
'<p>
The median is the middle most value of data set. 
</p>
'''

# Sentence 1
s221 = '''
<p>
If you want to find the Median of your data set, set all the data points within the data set in numerical order. 
</p>
'''

s222 = '''
<p>
In order to get the Median set the data points within the data set in numerical order.
</p>
'''

s223 = '''
<p>
You can get the Median of a data set by setting out all the data points in order. 
</p>
'''

s224 = '''
<p>
The Median is found by locating the midpoint within the data set. 
</p>
'''

s225 = '''
<p>
Order the values within the data set in numerical order. Look for the center value. 
</p>
'''

s226 = '''
<p>
Lay out the data in numerical order. 
</p>
'''

s227 = '''
<p>
Lay out the data set from least to greatest and locate the middle value. 
</p>
'''

# Sentence 2
s231 = '''
<p>
If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points. Then the Mean of the two mid points will be the median of the data set.
</p>
'''

s232 = '''
<p>
If your data set does not have a midpoint due to there being an even number set of data points. Find the mean of the two midpoints. 
</p>
'''

s233 = '''
<p>
When the data set has an even number of data points the Mean of the two most center points will be taken and that will be known as the median of the data set.
</p>
'''

s234 = '''
<p>
If the midpoint of your data set cannot be located due to the data having an odd number of data points. Then the two midpoints of the data set will be taken and the Mean of the two points will be found and become the median. 
</p>
'''

s235 = '''
<p>
If the data set does not have a center point. Find the two most midpoints within the data set and find their mean. That Mean is the Median of the data set. 
</p>
'''

s236 = '''
<p>
If the data has an even set of numbers and no center point then calculate the mean of the two center points.
</p>
'''

s237 = '''
<p>
If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points. Then the Mean of the two mid points will be the median of the data set.
</p>
'''

# Parargraph 2
# Mode
h = '''
<h3>
Mode
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''<p>The Mode of the dataset is the most reccuring data point through out the data set</p>'''

s312 = h + '''
<p>
The Value or frequency to appear within the data set most often is referred to as the Mode
</p>
'''

s313 = h + '''
<p>
The Mode is the value or data point that appears most often within the data set 
</p>
'''

s314 = h + '''
<p>
When a data set contains a value that occurs more than once it is known as the Mode. This is the Value most likely to be picked fro the data set
</p>
'''

s315 = h + '''
<p>
The value that appears most often is the Mode
</p>
'''

s316 = h + '''
<p>
The value that appears most frequently within the data set
</p>
'''

# Sentence 1

s321 = '''
<p>
One you have found the Median look for the value that is most recurring this the Mode
</p>
'''

s322 = '''
<p>
It is easiest to find the Mode once you have the Median. 
</p>
'''

s323 = '''
<p>
After you have found the Median of the data set it is much easier to locate which value occurs the most with the data points laid out. 
</p>
'''

s324 = '''
<p>
To Calculate the mode align the data points is numerical order and find the data point the occurs the most often.
</p>
'''

s325 = '''
<p>
The Mode is easiest to locate after the median is located. Once you have the data points laid out locate the value that recurrs the most often
</p>
'''

s326 = '''
<p>
Put the data points from least to greatest and count the value that recurrs the most
</p>
'''

# Sentence 2

s331 = '''
<p>
A dataset can contain more contain more than one mode. If there are only recurring values the data set is Bimodal. 
</p>
'''

s332 = '''
<p>
If the dataset has two recurring values then it is Bimodal
</p>
'''

s333 = '''
<p>
Often a data set may contain more than one recurring value. In such cases the data set is referred to as bimodal
</p>
'''

s334 = '''
<p>
A bimodal data set is a set that contains two Modal Values
</p>
'''

s335 = '''
<p>
A data set containing two modal values is a Bimodal data set
</p>
'''

s336 = '''
<p>
If two values recurr throughout the data an equal amount of times the data set is Bimodal
</p>
'''

# Sentence 3

s341 = '''
<p>
If the dataset has three recurring values then the data set is tri modal
</p>
'''

s342 = '''
<p>
When the data set has three recurring values it is Trimodal
</p>
'''

s343 = '''
<p>
If the data set contains three recurring values. The data set is trimodal
</p>
'''

s344 = '''
<p>
A trimodal data set is a data set that contains three modal values
</p>
'''

s345 = '''
<p>
A data set containing three modal values is a Trimodal
</p>
'''

s346 = '''
<p>
If three values recur throughout the data an equal amount of times the dataset is trimodal
</p>
'''

# Sentence 5

s351 = '''
<p>
If the dataset has more than three recurring values then the data set is multi modal
</p>
'''

s352 = '''
<p>
When the data set has above three recurring values. The data set is multi modal
</p>
'''

s353 = '''
<p>
In larger data sets you may see multiple recurring data sets, In such cases the data set is multi modal
</p>
'''

s354 = '''
<p>
A multi modal set is a dataset that contains multiple recurring values
</p>
'''

s355 = '''
<p>
A data set containing multiple values is Multimodal
</p>
'''

s356 = '''
<p>
If multiple values recur throughout the dataset the set is multimodal 
</p>
'''

# PARAGRAPH 3
# Variance

h = '''
<h3>
Variance
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
 <p>
The Variance is how far a data sets values are spread out from their average mean. 
</p>
'''

s412 = h + ''' 
<p>
The variance measures how far each data point is from the mean. 
</p>
'''

# Sentence 1
s421 = '''
<p>
To calculate the Variance take the Mean of the data set and subtract it from each data point. Square each result and find the sum of the squared values. Divide by the number of data points and subtract 1. 
</p>
'''

s422 = '''
<p>
The Variance is calculated through taking by subtracting the mean from each data point within the data set. Subtract it from each data point. Square the results and add it all together. take the sum and divide it by the number of data points and subtract it by 1.
</p>
'''

# Sentence 2
s431 = '''
<p>
Square the root of the Variance to calculate the Standard Deviation. 
</p>
'''

s432 = '''
<p>
The Standard Deviation of the data set is found by taking the Variance and squaring the root. 
</p>
#### 
'''

# Sentence 3
s441 = '''
<p>
Standard deviation is used to calculate the amount of variation within the data set. 
</p>
'''

s442 = '''
<p>
The Standard Deviation calculates the amount of variation within the data set. 
</p>
'''

# Paragraph 5
# Skew


h = '''
<h3>
Skew
</h3>
'''

s511 = h + '''
<p>
The Skew of a data set is the measure of the lack of symmetry within the data. The Skewness of the data can be either positive, negative, or undefined
</p>
'''

# Sentence 1
s512 = h + '''
<p>
Skewness describes the symmetry of a distribution. Or lack there of symmetry. If the data has a tail end distribution then the data is skewed. 
</p>
'''

s513 = h + '''
<p>
The Distribution of a data set is the Skewness of the data set. This defines whether the data is symmetrical or not
</p>
'''

s514 = h + '''
<p>
The skew defines the distribution of the data set. 
</p>
'''

s515 = h + '''
<p>
Skewness is when the curve of the data is distorted to either the left or the right
</p>
'''

# Sentence 1
s521 = '''
<p>

</p>
'''

s522 = '''
<p>

</p>
'''

s523 = '''
<p>

</p>
'''

s524 = '''
<p>

</p>
'''

s525 = '''
<p>

</p>
'''

# Sentence 2
s531 = '''
<p>
If the data is skewed to the right this is known as a positive distribution. In a positive distribution the mean usually greater than the median, both are to the right of the Mode.  
</p>
'''

s532 = '''
<p>
If the tail of the data extends to the right then the data is positively skewed. In a positively skewed distribution the Mean is usually to the right of the median and both are to the right of the Mode. 
</p>
'''

s533 = '''
<p>
When the tail of the data lays to the right of the curve. The data has a positive skew. In a positive skew the Mean is greater than the median and both are to the right of the mode. 
</p>
'''

s534 = '''
<p>
If the right side of the data has a tail then the data is positively skewed. The mean and the median lie to the right of the mode. 
</p>
'''

s535 = '''
<p>
A curve aligned to the right extending a tail to the left is a negatively skewed curve.
</p>
'''

# Sentence 3
s541 = '''
<p>
If the data is skewed to the left this is known as a negative disribution. In a negative distribution the Mean is less than the median and both are less than the mode. 
</p>
'''

s542 = '''
<p>
If the tail of the data extends to the left then the data is negatively skewed. In a negative distribution the Mean is to the left of the median and both are to the left of the mode. 
</p>
'''

s543 = '''
<p>
When the tail of the data lays to the left of the curve the mean is less than the median and both are to the left of the curve. 
</p>
'''

s544 = '''
<p>
If the left side of the data has a tail end. The data is negatively skewed. The mean is less than the median and the Mode. 
</p>
'''

s545 = '''
<p>
A curve aligned to the left extending a tail to the right is a positively skewed curve.
</p>
'''

# Sentence 5
s551 = '''
<p>
The distribution of the data is symmetrical when the data lacks skewness. 
</p>
'''

s552 = '''
<p>
When the data does not contain a tail it is a bell curve. In a bell curve the data is symmetrical. 
</p>
'''

s553 = '''
<p>
If there is no extending tail in the data the data is a Bell Curve. A symmetrical data set
</p>
'''

s554 = '''
<p>
A data set is symmetrical if it looks the same on the left and the right side.
</p>
'''

s555 = '''
<p>
A curve in the center of the data is a bell curve, a symmetrical curve.
</p>
'''

# Paragraph 6
# Kurtosis
# Sentence 1


h = '''
<h3>
Kurtosis
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s611 = h + '''
 <p>
 Kurtosis is the measure of sharpnesss in the peak of the distribution. If it is a positive, megaitve, or normal curve.
</p>
'''

s612 = h + '''
 <p>
The measure of the shape of the curve. Kurtosis measures if it is flat, normal, or peaked
</p>
'''

s613 = h + '''
 <p>
Kurtosis is the measure of the shape of the distribution. A measure of Kurtosis can either be a 0, a negative, or a positive number
</p>
'''

s614 = h + '''
 <p>
Kurtosis measures how flat or peaked out data distribution is.
</p>
'''

s615 = h + '''
<p>
Kurtosis refers to the peakedness or flatness of the distribution.
</p>
'''

# Sentence 1
s621 = '''
 <p>
When the data does not have a peak distribution but resembles a normal bell curve this is called Mesokurtic.
</p>
'''

s622 = '''
 <p>
When the curve has the bell curve of a normal distribution then it is a Mesokurtic Distribution
</p>
'''

s623 = '''
 <p>
When the calculation of Kurtosis is that of a normal distribution then the distribution is that of a Mesokurtic
</p>
'''

s624 = '''
<p>
When the measure of Kurtosis is a positive distribution it will have a extended peak. This is known as Mesokurtic.
</p>
'''

s625 = '''
<p>
When the distribution has a sharp peak this is a positive kurtosis known as a mesokurtic distribution
</p>
'''

# Sentence 2
s631 = '''
<p>
A negative kurtosis is a Mesokurtic distribution. This will have closer to what looks like a flat distribution.
</p>
'''

s632 = '''
<p>
When the curve is flatter than a normal bell distribution this is called a Platykurtic distribution
</p>
'''

s633 = '''
<p>
When the calculation of Kurtosis is a negative number this results in a Platykurtic distribution. A peak that is flatter than a normal distribution.
</p>
'''

s634 = '''
<p>
When the measure of Kurtosis has a negative distribution it will have a flatter lower curve this curve is known as Platykurtic.
</p>
'''

s635 = '''
<p>

</p>
'''
# Sentence 3
s641 = '''
<p>
A positive kurtosis is called Leptokurtic. In a Leptokurtic distribution the data has a sharp peak 
</p>
'''

s642 = '''
<p>
When the curve is taller and skinnier than a normal distribution it has a positive Kurtosis also known as a Letokurtic Distribution
</p>
'''

s643 = '''
<p>
When the calculation of Kurtosis is a positive number this results in a Leptokurtic distribution, or a sharp peak. 
</p>
'''

s644 = '''
<p>
When the measure of Kutosis has a regular distribution or a distribution of 0 it will resemble the normal bell ccurve, also known as a Leptokurtic distribution.</p>
'''

s645 = '''
<p>

</p>
'''

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
# MEAN
h = '''
<h3>
Mean
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
The Mean of your data set is also known as the average of your data set.
</p>
'''

s112 = h + '''
<p>
The Average of your data set is what is called the Mean or the central value of the data set.
</p>
'''

s113 = h + '''
<p>
The Mean is the average of all the data points within the data set and is sometime called the Arithmetic Mean.
</p>
'''

s114 = h + '''
<p>
One way to find the Central Tendency of your data is to find the Mean (average) of your data set. 
</p>
'''

s115 = h + '''
<p>
The Mean is the Average of your data set
</p>
'''

s116 = h + '''
<p>
The average of all the data points within the data set is called the Mean of the the data set. 
</p>
'''

s117 = h + '''
<p>
The mean is just another term for the average of the data set.
</p>
'''

# Sentence 1
s121 = '''
<p> 
To get the Mean you must add up all the data points. take the sum and divide by the total number of data points. 
</p>
'''

s122 = '''
<p>
The Mean is the summation of the data set points divided by the number of dataset points. 
</p>
'''

s123 = '''
<p>
To calculate the Average (Mean), add together all the data points within the data set, and then divide the sum by the total number of data points within the data set.
</p>
'''

s124 = '''
<p>
The average is calculated by first adding up all the data points within your data set. Take the sum and divide it by the total number of data points within youur set. 
</p>
'''

s125 = '''
<p>
You must first add up the data points within your data set and then divide by the total number of data points. 
</p>
'''

s126 = '''
<p>
The mean is calculated by adding up all the data points and dividing by the total number of data points. 
</p>
'''

s127 = '''
<p>
Add all the values together and divide them by the number of data points within the data set. 
</p>
'''

# Sentence 2
s131 = '''
<p>
By doing this you have a better understanding of what the average of your data set is.
</p>
'''

s132 = '''
####
<p>
The Mean is one way to find the Central Tendency of your data set. The central tendency can be used to define larger data sets by one value. 
</p>
'''

s133 = '''
<p>
Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number. 
</p>
'''

s134 = '''
####
<p>
Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number. 
</p>
'''

s135 = '''
####
<p>
This gives you a central number to define your data
</p>
'''

s136 = '''
####
<p>
This gives you a central number to define your data
</p>
'''

# Paragraph2
# MEDIAN

h = '''
<h3>
Median 
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

# Sentence 1
s211 = h + '''
<p>
The Median is the midpoint of the data set. It is the point that separates the lower half of your data from the top half.
</p>
'''

s212 = h + '''
<p>
The midpoint of the data set is known as the Median. The Median is used to represent the center of your data set.
</p>
'''

s213 = h + '''
<p>
Your data can be split into two sets, a lower half and a top half. The midpoint of the set is known as the Median
\</p>
'''

s214 = h + '''
'<p>
The Median is the data point lying at the midpoint of the data set.
</p>
'''

s215 = h + '''
'<p>
The Median is the center value within a data set.
</p>
'''

s216 = h + '''
'<p>
The median is the middle value of the data set
</p>
'''

s217 = h + '''
'<p>
The median is the middle most value of data set. 
</p>
'''

# Sentence 1
s221 = '''
<p>
If you want to find the Median of your data set, set all the data points within the data set in numerical order. 
</p>
'''

s222 = '''
<p>
In order to get the Median set the data points within the data set in numerical order.
</p>
'''

s223 = '''
<p>
You can get the Median of a data set by setting out all the data points in order. 
</p>
'''

s224 = '''
<p>
The Median is found by locating the midpoint within the data set. 
</p>
'''

s225 = '''
<p>
Order the values within the data set in numerical order. Look for the center value. 
</p>
'''

s226 = '''
<p>
Lay out the data in numerical order. 
</p>
'''

s227 = '''
<p>
Lay out the data set from least to greatest and locate the middle value. 
</p>
'''

# Sentence 2
s231 = '''
<p>
If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points. Then the Mean of the two mid points will be the median of the data set.
</p>
'''

s232 = '''
<p>
If your data set does not have a midpoint due to there being an even number set of data points. Find the mean of the two midpoints. 
</p>
'''

s233 = '''
<p>
When the data set has an even number of data points the Mean of the two most center points will be taken and that will be known as the median of the data set.
</p>
'''

s234 = '''
<p>
If the midpoint of your data set cannot be located due to the data having an odd number of data points. Then the two midpoints of the data set will be taken and the Mean of the two points will be found and become the median. 
</p>
'''

s235 = '''
<p>
If the data set does not have a center point. Find the two most midpoints within the data set and find their mean. That Mean is the Median of the data set. 
</p>
'''

s236 = '''
<p>
If the data has an even set of numbers and no center point then calculate the mean of the two center points.
</p>
'''

s237 = '''
<p>
If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points. Then the Mean of the two mid points will be the median of the data set.
</p>
'''

# Parargraph 2
# Mode
h = '''
<h3>
Mode
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''<p>The Mode of the dataset is the most reccuring data point through out the data set</p>'''

s312 = h + '''
<p>
The Value or frequency to appear within the data set most often is referred to as the Mode
</p>
'''

s313 = h + '''
<p>
The Mode is the value or data point that appears most often within the data set 
</p>
'''

s314 = h + '''
<p>
When a data set contains a value that occurs more than once it is known as the Mode. This is the Value most likely to be picked fro the data set
</p>
'''

s315 = h + '''
<p>
The value that appears most often is the Mode
</p>
'''

s316 = h + '''
<p>
The value that appears most frequently within the data set
</p>
'''

# Sentence 1

s321 = '''
<p>
One you have found the Median look for the value that is most recurring this the Mode
</p>
'''

s322 = '''
<p>
It is easiest to find the Mode once you have the Median. 
</p>
'''

s323 = '''
<p>
After you have found the Median of the data set it is much easier to locate which value occurs the most with the data points laid out. 
</p>
'''

s324 = '''
<p>
To Calculate the mode align the data points is numerical order and find the data point the occurs the most often.
</p>
'''

s325 = '''
<p>
The Mode is easiest to locate after the median is located. Once you have the data points laid out locate the value that recurrs the most often
</p>
'''

s326 = '''
<p>
Put the data points from least to greatest and count the value that recurrs the most
</p>
'''

# Sentence 2

s331 = '''
<p>
A dataset can contain more contain more than one mode. If there are only recurring values the data set is Bimodal. 
</p>
'''

s332 = '''
<p>
If the dataset has two recurring values then it is Bimodal
</p>
'''

s333 = '''
<p>
Often a data set may contain more than one recurring value. In such cases the data set is referred to as bimodal
</p>
'''

s334 = '''
<p>
A bimodal data set is a set that contains two Modal Values
</p>
'''

s335 = '''
<p>
A data set containing two modal values is a Bimodal data set
</p>
'''

s336 = '''
<p>
If two values recurr throughout the data an equal amount of times the data set is Bimodal
</p>
'''

# Sentence 3

s341 = '''
<p>
If the dataset has three recurring values then the data set is tri modal
</p>
'''

s342 = '''
<p>
When the data set has three recurring values it is Trimodal
</p>
'''

s343 = '''
<p>
If the data set contains three recurring values. The data set is trimodal
</p>
'''

s344 = '''
<p>
A trimodal data set is a data set that contains three modal values
</p>
'''

s345 = '''
<p>
A data set containing three modal values is a Trimodal
</p>
'''

s346 = '''
<p>
If three values recur throughout the data an equal amount of times the dataset is trimodal
</p>
'''

# Sentence 5

s351 = '''
<p>
If the dataset has more than three recurring values then the data set is multi modal
</p>
'''

s352 = '''
<p>
When the data set has above three recurring values. The data set is multi modal
</p>
'''

s353 = '''
<p>
In larger data sets you may see multiple recurring data sets, In such cases the data set is multi modal
</p>
'''

s354 = '''
<p>
A multi modal set is a dataset that contains multiple recurring values
</p>
'''

s355 = '''
<p>
A data set containing multiple values is Multimodal
</p>
'''

s356 = '''
<p>
If multiple values recur throughout the dataset the set is multimodal 
</p>
'''

# PARAGRAPH 3
# Variance

h = '''
<h3>
Variance
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
 <p>
The Variance is how far a data sets values are spread out from their average mean. 
</p>
'''

s412 = h + ''' 
<p>
The variance measures how far each data point is from the mean. 
</p>
'''

# Sentence 1
s421 = '''
<p>
To calculate the Variance take the Mean of the data set and subtract it from each data point. Square each result and find the sum of the squared values. Divide by the number of data points and subtract 1. 
</p>
'''

s422 = '''
<p>
The Variance is calculated through taking by subtracting the mean from each data point within the data set. Subtract it from each data point. Square the results and add it all together. take the sum and divide it by the number of data points and subtract it by 1.
</p>
'''

# Sentence 2
s431 = '''
<p>
Square the root of the Variance to calculate the Standard Deviation. 
</p>
'''

s432 = '''
<p>
The Standard Deviation of the data set is found by taking the Variance and squaring the root. 
</p>
#### 
'''

# Sentence 3
s441 = '''
<p>
Standard deviation is used to calculate the amount of variation within the data set. 
</p>
'''

s442 = '''
<p>
The Standard Deviation calculates the amount of variation within the data set. 
</p>
'''

# Paragraph 5
# Skew


h = '''
<h3>
Skew
</h3>
'''

s511 = h + '''
<p>
The Skew of a data set is the measure of the lack of symmetry within the data. The Skewness of the data can be either positive, negative, or undefined
</p>
'''

# Sentence 1
s512 = h + '''
<p>
Skewness describes the symmetry of a distribution. Or lack there of symmetry. If the data has a tail end distribution then the data is skewed. 
</p>
'''

s513 = h + '''
<p>
The Distribution of a data set is the Skewness of the data set. This defines whether the data is symmetrical or not
</p>
'''

s514 = h + '''
<p>
The skew defines the distribution of the data set. 
</p>
'''

s515 = h + '''
<p>
Skewness is when the curve of the data is distorted to either the left or the right
</p>
'''

# Sentence 1
s521 = '''
<p>

</p>
'''

s522 = '''
<p>

</p>
'''

s523 = '''
<p>

</p>
'''

s524 = '''
<p>

</p>
'''

s525 = '''
<p>

</p>
'''

# Sentence 2
s531 = '''
<p>
If the data is skewed to the right this is known as a positive distribution. In a positive distribution the mean usually greater than the median, both are to the right of the Mode.  
</p>
'''

s532 = '''
<p>
If the tail of the data extends to the right then the data is positively skewed. In a positively skewed distribution the Mean is usually to the right of the median and both are to the right of the Mode. 
</p>
'''

s533 = '''
<p>
When the tail of the data lays to the right of the curve. The data has a positive skew. In a positive skew the Mean is greater than the median and both are to the right of the mode. 
</p>
'''

s534 = '''
<p>
If the right side of the data has a tail then the data is positively skewed. The mean and the median lie to the right of the mode. 
</p>
'''

s535 = '''
<p>
A curve aligned to the right extending a tail to the left is a negatively skewed curve.
</p>
'''

# Sentence 3
s541 = '''
<p>
If the data is skewed to the left this is known as a negative disribution. In a negative distribution the Mean is less than the median and both are less than the mode. 
</p>
'''

s542 = '''
<p>
If the tail of the data extends to the left then the data is negatively skewed. In a negative distribution the Mean is to the left of the median and both are to the left of the mode. 
</p>
'''

s543 = '''
<p>
When the tail of the data lays to the left of the curve the mean is less than the median and both are to the left of the curve. 
</p>
'''

s544 = '''
<p>
If the left side of the data has a tail end. The data is negatively skewed. The mean is less than the median and the Mode. 
</p>
'''

s545 = '''
<p>
A curve aligned to the left extending a tail to the right is a positively skewed curve.
</p>
'''

# Sentence 5
s551 = '''
<p>
The distribution of the data is symmetrical when the data lacks skewness. 
</p>
'''

s552 = '''
<p>
When the data does not contain a tail it is a bell curve. In a bell curve the data is symmetrical. 
</p>
'''

s553 = '''
<p>
If there is no extending tail in the data the data is a Bell Curve. A symmetrical data set
</p>
'''

s554 = '''
<p>
A data set is symmetrical if it looks the same on the left and the right side.
</p>
'''

s555 = '''
<p>
A curve in the center of the data is a bell curve, a symmetrical curve.
</p>
'''

# Paragraph 6
# Kurtosis
# Sentence 1


h = '''
<h3>
Kurtosis
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s611 = h + '''
 <p>
 Kurtosis is the measure of sharpnesss in the peak of the distribution. If it is a positive, megaitve, or normal curve.
</p>
'''

s612 = h + '''
 <p>
The measure of the shape of the curve. Kurtosis measures if it is flat, normal, or peaked
</p>
'''

s613 = h + '''
 <p>
Kurtosis is the measure of the shape of the distribution. A measure of Kurtosis can either be a 0, a negative, or a positive number
</p>
'''

s614 = h + '''
 <p>
Kurtosis measures how flat or peaked out data distribution is.
</p>
'''

s615 = h + '''
<p>
Kurtosis refers to the peakedness or flatness of the distribution.
</p>
'''

# Sentence 1
s621 = '''
 <p>
When the data does not have a peak distribution but resembles a normal bell curve this is called Mesokurtic.
</p>
'''

s622 = '''
 <p>
When the curve has the bell curve of a normal distribution then it is a Mesokurtic Distribution
</p>
'''

s623 = '''
 <p>
When the calculation of Kurtosis is that of a normal distribution then the distribution is that of a Mesokurtic
</p>
'''

s624 = '''
<p>
When the measure of Kurtosis is a positive distribution it will have a extended peak. This is known as Mesokurtic.
</p>
'''

s625 = '''
<p>
When the distribution has a sharp peak this is a positive kurtosis known as a mesokurtic distribution
</p>
'''

# Sentence 2
s631 = '''
<p>
A negative kurtosis is a Mesokurtic distribution. This will have closer to what looks like a flat distribution.
</p>
'''

s632 = '''
<p>
When the curve is flatter than a normal bell distribution this is called a Platykurtic distribution
</p>
'''

s633 = '''
<p>
When the calculation of Kurtosis is a negative number this results in a Platykurtic distribution. A peak that is flatter than a normal distribution.
</p>
'''

s634 = '''
<p>
When the measure of Kurtosis has a negative distribution it will have a flatter lower curve this curve is known as Platykurtic.
</p>
'''

s635 = '''
<p>

</p>
'''
# Sentence 3
s641 = '''
<p>
A positive kurtosis is called Leptokurtic. In a Leptokurtic distribution the data has a sharp peak 
</p>
'''

s642 = '''
<p>
When the curve is taller and skinnier than a normal distribution it has a positive Kurtosis also known as a Letokurtic Distribution
</p>
'''

s643 = '''
<p>
When the calculation of Kurtosis is a positive number this results in a Leptokurtic distribution, or a sharp peak. 
</p>
'''

s644 = '''
<p>
When the measure of Kutosis has a regular distribution or a distribution of 0 it will resemble the normal bell ccurve, also known as a Leptokurtic distribution.</p>
'''

s645 = '''
<p>

</p>
'''

paragraphs = [3, 3, 5, 4, 5, 4]
variations = 2
title = 'Descriptive Statistics'
abstract = True

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
     'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
     'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

# PARAGRAPH 1
# Sentence 1
# MEAN
h = '''
<h3>
Mean
</h3>
'''

img = '''
<image src = ''' + mae + ''' object_oriented="functionString">
'''

s111 = h + '''
<p>
The Mean of your data set is also known as the average of your data set.
</p>
'''

s112 = h + '''
<p>
The Average of your data set is what is called the Mean or the central value of the data set.
</p>
'''

s113 = h + '''
<p>
The Mean is the average of all the data points within the data set and is sometime called the Arithmetic Mean.
</p>
'''

s114 = h + '''
<p>
One way to find the Central Tendency of your data is to find the Mean (average) of your data set. 
</p>
'''

s115 = h + '''
<p>
The Mean is the Average of your data set
</p>
'''

s116 = h + '''
<p>
The average of all the data points within the data set is called the Mean of the the data set. 
</p>
'''

s117 = h + '''
<p>
The mean is just another term for the average of the data set.
</p>
'''

# Sentence 1
s121 = '''
<p> 
To get the Mean you must add up all the data points. take the sum and divide by the total number of data points. 
</p>
'''

s122 = '''
<p>
The Mean is the summation of the data set points divided by the number of dataset points. 
</p>
'''

s123 = '''
<p>
To calculate the Average (Mean), add together all the data points within the data set, and then divide the sum by the total number of data points within the data set.
</p>
'''

s124 = '''
<p>
The average is calculated by first adding up all the data points within your data set. Take the sum and divide it by the total number of data points within youur set. 
</p>
'''

s125 = '''
<p>
You must first add up the data points within your data set and then divide by the total number of data points. 
</p>
'''

s126 = '''
<p>
The mean is calculated by adding up all the data points and dividing by the total number of data points. 
</p>
'''

s127 = '''
<p>
Add all the values together and divide them by the number of data points within the data set. 
</p>
'''

# Sentence 2
s131 = '''
<p>
By doing this you have a better understanding of what the average of your data set is.
</p>
'''

s132 = '''
####
<p>
The Mean is one way to find the Central Tendency of your data set. The central tendency can be used to define larger data sets by one value. 
</p>
'''

s133 = '''
<p>
Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number. 
</p>
'''

s134 = '''
####
<p>
Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number. 
</p>
'''

s135 = '''
####
<p>
This gives you a central number to define your data
</p>
'''

s136 = '''
####
<p>
This gives you a central number to define your data
</p>
'''

# Paragraph2
# MEDIAN

h = '''
<h3>
Median 
</h3>
'''

img = '''
<image src = ''' + mbe + ''' object_oriented="functionString">
'''

# Sentence 1
s211 = h + '''
<p>
The Median is the midpoint of the data set. It is the point that separates the lower half of your data from the top half.
</p>
'''

s212 = h + '''
<p>
The midpoint of the data set is known as the Median. The Median is used to represent the center of your data set.
</p>
'''

s213 = h + '''
<p>
Your data can be split into two sets, a lower half and a top half. The midpoint of the set is known as the Median
\</p>
'''

s214 = h + '''
'<p>
The Median is the data point lying at the midpoint of the data set.
</p>
'''

s215 = h + '''
'<p>
The Median is the center value within a data set.
</p>
'''

s216 = h + '''
'<p>
The median is the middle value of the data set
</p>
'''

s217 = h + '''
'<p>
The median is the middle most value of data set. 
</p>
'''

# Sentence 1
s221 = '''
<p>
If you want to find the Median of your data set, set all the data points within the data set in numerical order. 
</p>
'''

s222 = '''
<p>
In order to get the Median set the data points within the data set in numerical order.
</p>
'''

s223 = '''
<p>
You can get the Median of a data set by setting out all the data points in order. 
</p>
'''

s224 = '''
<p>
The Median is found by locating the midpoint within the data set. 
</p>
'''

s225 = '''
<p>
Order the values within the data set in numerical order. Look for the center value. 
</p>
'''

s226 = '''
<p>
Lay out the data in numerical order. 
</p>
'''

s227 = '''
<p>
Lay out the data set from least to greatest and locate the middle value. 
</p>
'''

# Sentence 2
s231 = '''
<p>
If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points. Then the Mean of the two mid points will be the median of the data set.
</p>
'''

s232 = '''
<p>
If your data set does not have a midpoint due to there being an even number set of data points. Find the mean of the two midpoints. 
</p>
'''

s233 = '''
<p>
When the data set has an even number of data points the Mean of the two most center points will be taken and that will be known as the median of the data set.
</p>
'''

s234 = '''
<p>
If the midpoint of your data set cannot be located due to the data having an odd number of data points. Then the two midpoints of the data set will be taken and the Mean of the two points will be found and become the median. 
</p>
'''

s235 = '''
<p>
If the data set does not have a center point. Find the two most midpoints within the data set and find their mean. That Mean is the Median of the data set. 
</p>
'''

s236 = '''
<p>
If the data has an even set of numbers and no center point then calculate the mean of the two center points.
</p>
'''

s237 = '''
<p>
If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points. Then the Mean of the two mid points will be the median of the data set.
</p>
'''

# Parargraph 2
# Mode
h = '''
<h3>
Mode
</h3>'''

img = '''
<image src = ''' + mse + ''' object_oriented="functionString">
'''

s311 = h + '''<p>The Mode of the dataset is the most reccuring data point through out the data set</p>'''

s312 = h + '''
<p>
The Value or frequency to appear within the data set most often is referred to as the Mode
</p>
'''

s313 = h + '''
<p>
The Mode is the value or data point that appears most often within the data set 
</p>
'''

s314 = h + '''
<p>
When a data set contains a value that occurs more than once it is known as the Mode. This is the Value most likely to be picked fro the data set
</p>
'''

s315 = h + '''
<p>
The value that appears most often is the Mode
</p>
'''

s316 = h + '''
<p>
The value that appears most frequently within the data set
</p>
'''

# Sentence 1

s321 = '''
<p>
One you have found the Median look for the value that is most recurring this the Mode
</p>
'''

s322 = '''
<p>
It is easiest to find the Mode once you have the Median. 
</p>
'''

s323 = '''
<p>
After you have found the Median of the data set it is much easier to locate which value occurs the most with the data points laid out. 
</p>
'''

s324 = '''
<p>
To Calculate the mode align the data points is numerical order and find the data point the occurs the most often.
</p>
'''

s325 = '''
<p>
The Mode is easiest to locate after the median is located. Once you have the data points laid out locate the value that recurrs the most often
</p>
'''

s326 = '''
<p>
Put the data points from least to greatest and count the value that recurrs the most
</p>
'''

# Sentence 2

s331 = '''
<p>
A dataset can contain more contain more than one mode. If there are only recurring values the data set is Bimodal. 
</p>
'''

s332 = '''
<p>
If the dataset has two recurring values then it is Bimodal
</p>
'''

s333 = '''
<p>
Often a data set may contain more than one recurring value. In such cases the data set is referred to as bimodal
</p>
'''

s334 = '''
<p>
A bimodal data set is a set that contains two Modal Values
</p>
'''

s335 = '''
<p>
A data set containing two modal values is a Bimodal data set
</p>
'''

s336 = '''
<p>
If two values recurr throughout the data an equal amount of times the data set is Bimodal
</p>
'''

# Sentence 3

s341 = '''
<p>
If the dataset has three recurring values then the data set is tri modal
</p>
'''

s342 = '''
<p>
When the data set has three recurring values it is Trimodal
</p>
'''

s343 = '''
<p>
If the data set contains three recurring values. The data set is trimodal
</p>
'''

s344 = '''
<p>
A trimodal data set is a data set that contains three modal values
</p>
'''

s345 = '''
<p>
A data set containing three modal values is a Trimodal
</p>
'''

s346 = '''
<p>
If three values recur throughout the data an equal amount of times the dataset is trimodal
</p>
'''

# Sentence 5

s351 = '''
<p>
If the dataset has more than three recurring values then the data set is multi modal
</p>
'''

s352 = '''
<p>
When the data set has above three recurring values. The data set is multi modal
</p>
'''

s353 = '''
<p>
In larger data sets you may see multiple recurring data sets, In such cases the data set is multi modal
</p>
'''

s354 = '''
<p>
A multi modal set is a dataset that contains multiple recurring values
</p>
'''

s355 = '''
<p>
A data set containing multiple values is Multimodal
</p>
'''

s356 = '''
<p>
If multiple values recur throughout the dataset the set is multimodal 
</p>
'''

# PARAGRAPH 3
# Variance

h = '''
<h3>
Variance
</h3>
'''

img = '''
<image src = ''' + rmse + ''' object_oriented="functionString">
'''

s411 = h + '''
 <p>
The Variance is how far a data sets values are spread out from their average mean. 
</p>
'''

s412 = h + ''' 
<p>
The variance measures how far each data point is from the mean. 
</p>
'''

# Sentence 1
s421 = '''
<p>
To calculate the Variance take the Mean of the data set and subtract it from each data point. Square each result and find the sum of the squared values. Divide by the number of data points and subtract 1. 
</p>
'''

s422 = '''
<p>
The Variance is calculated through taking by subtracting the mean from each data point within the data set. Subtract it from each data point. Square the results and add it all together. take the sum and divide it by the number of data points and subtract it by 1.
</p>
'''

# Sentence 2
s431 = '''
<p>
Square the root of the Variance to calculate the Standard Deviation. 
</p>
'''

s432 = '''
<p>
The Standard Deviation of the data set is found by taking the Variance and squaring the root. 
</p>
#### 
'''

# Sentence 3
s441 = '''
<p>
Standard deviation is used to calculate the amount of variation within the data set. 
</p>
'''

s442 = '''
<p>
The Standard Deviation calculates the amount of variation within the data set. 
</p>
'''

# Paragraph 5
# Skew


h = '''
<h3>
Skew
</h3>
'''

s511 = h + '''
<p>
The Skew of a data set is the measure of the lack of symmetry within the data. The Skewness of the data can be either positive, negative, or undefined
</p>
'''

# Sentence 1
s512 = h + '''
<p>
Skewness describes the symmetry of a distribution. Or lack there of symmetry. If the data has a tail end distribution then the data is skewed. 
</p>
'''

s513 = h + '''
<p>
The Distribution of a data set is the Skewness of the data set. This defines whether the data is symmetrical or not
</p>
'''

s514 = h + '''
<p>
The skew defines the distribution of the data set. 
</p>
'''

s515 = h + '''
<p>
Skewness is when the curve of the data is distorted to either the left or the right
</p>
'''

# Sentence 1
s521 = '''
<p>

</p>
'''

s522 = '''
<p>

</p>
'''

s523 = '''
<p>

</p>
'''

s524 = '''
<p>

</p>
'''

s525 = '''
<p>

</p>
'''

# Sentence 2
s531 = '''
<p>
If the data is skewed to the right this is known as a positive distribution. In a positive distribution the mean usually greater than the median, both are to the right of the Mode.  
</p>
'''

s532 = '''
<p>
If the tail of the data extends to the right then the data is positively skewed. In a positively skewed distribution the Mean is usually to the right of the median and both are to the right of the Mode. 
</p>
'''

s533 = '''
<p>
When the tail of the data lays to the right of the curve. The data has a positive skew. In a positive skew the Mean is greater than the median and both are to the right of the mode. 
</p>
'''

s534 = '''
<p>
If the right side of the data has a tail then the data is positively skewed. The mean and the median lie to the right of the mode. 
</p>
'''

s535 = '''
<p>
A curve aligned to the right extending a tail to the left is a negatively skewed curve.
</p>
'''

# Sentence 3
s541 = '''
<p>
If the data is skewed to the left this is known as a negative disribution. In a negative distribution the Mean is less than the median and both are less than the mode. 
</p>
'''

s542 = '''
<p>
If the tail of the data extends to the left then the data is negatively skewed. In a negative distribution the Mean is to the left of the median and both are to the left of the mode. 
</p>
'''

s543 = '''
<p>
When the tail of the data lays to the left of the curve the mean is less than the median and both are to the left of the curve. 
</p>
'''

s544 = '''
<p>
If the left side of the data has a tail end. The data is negatively skewed. The mean is less than the median and the Mode. 
</p>
'''

s545 = '''
<p>
A curve aligned to the left extending a tail to the right is a positively skewed curve.
</p>
'''

# Sentence 5
s551 = '''
<p>
The distribution of the data is symmetrical when the data lacks skewness. 
</p>
'''

s552 = '''
<p>
When the data does not contain a tail it is a bell curve. In a bell curve the data is symmetrical. 
</p>
'''

s553 = '''
<p>
If there is no extending tail in the data the data is a Bell Curve. A symmetrical data set
</p>
'''

s554 = '''
<p>
A data set is symmetrical if it looks the same on the left and the right side.
</p>
'''

s555 = '''
<p>
A curve in the center of the data is a bell curve, a symmetrical curve.
</p>
'''

paragraphs = [1, 1, 1, 1, 1, 1, 1]
variations = 1
title = 'Descriptive Statistics'
abstract = False

"""

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
           'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
            'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

"""

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean (Average)
</h3>
'''

s111 = h + '''
<p>
The Mean (average) of a data set is the sum of the combined data points divided by
the total amount of data points. 
</p>
'''

s112 = h + '''
<p>
The sum of your data points divided by the total number of data points is known as 
your Mean (average)
</p>
'''

s113 = h + '''
<p>
In order to obtain the average (Mean) of your data set, you must divide the sum of your
data points. 
</p>
'''

s114 = h + '''
<p>
The Average (Mean) of your data set is the sum of your data points divided by the total 
number of data points within the data set.
</p>
'''

# Median

h2 = '''
<h3> 
Median
</h3>
'''

s211 = h2 + '''
<p> 
The Median is the Midpoint of the data set.
</p>
'''

# Sentence 2

s212 = h2 + '''
<p>
It is the point that seperates the lower half of the 
data set from the top half.
</p>
'''

# Sentence 3
s221 = h2 + '''
<p>
When the data set has an even number of data points the
mean(average) of the two mid points within the data set will be taken
</p>
'''

# Mode

h3 = '''
<p>
<h3>Mode<h3>
</p>
'''

s311 = h3 + '''
<p>
The Mode of the data set is the most recurring value within the data set.
</p>
'''

s312 = h3 + '''
<p>
Within your data set there may be recurring values, these values are known
as the Mode of the dataset
</p>
'''

# Sentence2

s121 = h3 + '''
<p>
A data set can contain more than one Mode, if there are two Modes within 
the data set the data set is Bimodal.
</p>
'''

s122 = h3 + '''
<p>
It is common to have more than one Mode within a data set, a data set with
two recurring values is Bimodal
</p>
'''

# Sentence 3

s131 = h3 + '''
<p>
It is not uncommon for data sets to contain multiple recurring values,
in such cases the data set is Multi-Modal
</p>
'''

# Variance

h4 = '''
<p> 
<h3>Variance<h3>
</p>
'''

s111 = h4 + '''
<p> 
The Variance is the spread of your data.
</p>
'''

s112 = h4 + '''
<p>

</p>
'''

paragraphs = [1, 3, 3, 1]
variations = 1
title = 'Descriptive Statistics'
abstract = False

"""

# ABSTRACT
a1 = '<p>Error metrics compare the Regression6 analysis to the data and ' \
           'are used to quantify the uncertainty or "error" in the Regression6 analysis.' \
     'The following are all of the error metrics included in this report.</p>'
a2 = '<p>Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis ' \
            'by comparing the Regression6 analysis with respect to the data.' \
     'This report includes all of the following error metrics.</p>'

"""

# PARAGRAPH 1
# Sentence 1
h = '''
<h3>
Mean (Average)
</h3>
'''

s111 = h + '''
<p>
The Mean (average) of a data set is the sum of the combined data points divided by
the total amount of data points. 
</p>
'''

s112 = h + '''
<p>
The sum of your data points divided by the total number of data points is known as 
your Mean (average)
</p>
'''

s113 = h + '''
<p>
In order to obtain the average (Mean) of your data set, you must divide the sum of your
data points. 
</p>
'''

s114 = h + '''
<p>
The Average (Mean) of your data set is the sum of your data points divided by the total 
number of data points within the data set.
</p>
'''

# Median

h2 = '''
<h3> 
Median
</h3>
'''

s211 = h2 + '''
<p> 
The Median is the Midpoint of the data set.
</p>
'''

# Sentence 1

s221 = '''
<p>
It is the point that seperates the lower half of the 
data set from the top half.
</p>
'''

# Sentence 2
s231 = '''
<p>
When the data set has an even number of data points the
mean(average) of the two mid points within the data set will be taken
</p>
'''

# Mode

h3 = '''
<h3>Mode</h3>
'''

s311 = h3 + '''
<p>
The Mode of the data set is the most recurring value within the data set.
</p>
'''

s312 = h3 + '''
<p>
Within your data set there may be recurring values, these values are known
as the Mode of the dataset
</p>
'''

# Sentence2

s321 = '''
<p>
A data set can contain more than one Mode, if there are two Modes within 
the data set the data set is Bimodal.
</p>
'''

s322 = '''
<p>
It is common to have more than one Mode within a data set, a data set with
two recurring values is Bimodal
</p>
'''

# Sentence 2

s331 = '''
<p>
It is not uncommon for data sets to contain multiple recurring values,
in such cases the data set is Multi-Modal
</p>
'''

# Variance

h4 = '''
<h3>Variance</h3>
'''

s411 = h4 + '''
<p> 
The Variance is the spread of your data.
</p>
'''

s412 = h4 + '''
<p>

</p>
'''

"""

def write_chapter(jobDir=None, jobName=None, templatePath='', **kwargs):

    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Sections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections contain content of Figure, Caption, or Paragraph)


    outputPath = os.path.join(jobDir,
                              jobName,
                              'InterQuartileRange.html')

    chapterTitle = 'InterQuartileRange'

    templateVars = {'chapterTitle':
                    chapterTitle,
                'chapterAbstract':
                    ReportGenerator.generate_textstruct(chapterAbstract),
                'chapterSections':
                    ReportGenerator.generate_sentences(chapterSections),
                'heavyBreak':
                    False}
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath,
                                          templateVars=templateVars,
                                          outputPath=outputPath)



def first_quartile_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>1st quartile</var> is the <em>50th Percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 25% of the data exists, and above which 75% of the data exists.'
    ]

    return [p0]

def second_quarile_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>2nd quartile</var> is the <em>50th percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 50% of the data is above and 50% is below'
    ]
    p0['sentence 1'] = [
        'This value is more commonly known as the <var>Median</var>.'
    ]

    return [p0]


def third_quartile_range_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>3rd quartile</var> is the <em>75th Percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 75% of the data exists, and above which 25% of the data exists.'
    ]

    return [p0]

def interquartile_range_desccription():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <em>IQR</em> is used to calculate <var>outliers</var>.'
    ]
    p0['sentence 1'] = [
        'Any value that is a distance of <var1.5*IQR</var><em>below the 1st Quartile or above the 3rd Quartile</em> is considered an outlier.'
    ]
    p0['sentence 1'] = [
        'This changes what how the minimum and maximum values are defined'
    ]
    p0['sentence 2'] = [
        'The IQR is calculated as Follows:'
    ]

    return [p0]


def first_quartile_section():
    firstQuartile = ReportGenerator.Section('First Quartile')
    firstQuartileDescription = ReportGenerator.Subsection('Description')
    firstQuartileDescription.insert_content(boxPlot_description())
    firstQuartileExample = ReportGenerator.Subsection('Example')
    firstQuartile.insert_subsections([firstQuartileDescription,
                             firstQuartileExample])
    return firstQuartile


def second_quartile_section():
    secondQuartile = ReportGenerator.Section('Second Quartile')
    secondQuartileDescription = ReportGenerator.Subsection('Description')
    secondQuartileExample = ReportGenerator.Subsection('Example')
    secondQuartile.insert_subsections([secondQuartileDescription,
                               secondQuartileExample])
    return secondQuartile


def third_quartile_section():
    thirdQuartile = ReportGenerator.Section('Third Quartile')
    thirdQuatrileDescription = ReportGenerator.Subsection('Description')
    thirdQuartileExample = ReportGenerator.Subsection('Example')
    thirdQUartile.insert_subsections([thirdQuartileDescription,
                             thirdQuartileExample])
    return thirdQuartile


def interquartile_range_section():
    iqr = ReportGenerator.Section('IQR')
    iqrDescription = ReportGenerator.Subsection('Description')
    iqrExample = ReportGenerator.Subsection('Example')
    iqr.insert_subsections([iqrDescription,
                             iqrExample])
    return iqr

def write_chapter():
    chapter = ReportGenerator.Chapter('Quartiles')
    firstQuartile = first_quartile_section()
    secondQuartile = second_quartile_section()
    thirdQuartile = third_quartile_section()
    IQR =  interquartile_range_section()
    chapter.insert_sections([firstQuartile])
    print(chapter)

write_chapter()

"""

"""

def write_chapter(jobDir=None, jobName=None, templatePath='', **kwargs):

    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Sections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections contain content of Figure, Caption, or Paragraph)


    outputPath = os.path.join(jobDir,
                              jobName,
                              'InterQuartileRange.html')

    chapterTitle = 'InterQuartileRange'

    templateVars = {'chapterTitle':
                    chapterTitle,
                'chapterAbstract':
                    ReportGenerator.generate_textstruct(chapterAbstract),
                'chapterSections':
                    ReportGenerator.generate_sentences(chapterSections),
                'heavyBreak':
                    False}
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath,
                                          templateVars=templateVars,
                                          outputPath=outputPath)



def first_quartile_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>1st quartile</var> is the <em>50th Percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 25% of the data exists, and above which 75% of the data exists.'
    ]

    return [p0]

def second_quarile_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>2nd quartile</var> is the <em>50th percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 50% of the data is above and 50% is below'
    ]
    p0['sentence 1'] = [
        'This value is more commonly known as the <var>Median</var>.'
    ]

    return [p0]


def third_quartile_range_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>3rd quartile</var> is the <em>75th Percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 75% of the data exists, and above which 25% of the data exists.'
    ]

    return [p0]

def interquartile_range_desccription():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <em>IQR</em> is used to calculate <var>outliers</var>.'
    ]
    p0['sentence 1'] = [
        'Any value that is a distance of <var1.5*IQR</var><em>below the 1st Quartile or above the 3rd Quartile</em> is considered an outlier.'
    ]
    p0['sentence 1'] = [
        'This changes what how the minimum and maximum values are defined'
    ]
    p0['sentence 2'] = [
        'The IQR is calculated as Follows:'
    ]

    return [p0]


def first_quartile_section():
    firstQuartile = ReportGenerator.Section('First Quartile')
    firstQuartileDescription = ReportGenerator.Subsection('Description')
    firstQuartileDescription.insert_content(boxPlot_description())
    firstQuartileExample = ReportGenerator.Subsection('Example')
    firstQuartile.insert_subsections([firstQuartileDescription,
                             firstQuartileExample])
    return firstQuartile


def second_quartile_section():
    secondQuartile = ReportGenerator.Section('Second Quartile')
    secondQuartileDescription = ReportGenerator.Subsection('Description')
    secondQuartileExample = ReportGenerator.Subsection('Example')
    secondQuartile.insert_subsections([secondQuartileDescription,
                               secondQuartileExample])
    return secondQuartile


def third_quartile_section():
    thirdQuartile = ReportGenerator.Section('Third Quartile')
    thirdQuatrileDescription = ReportGenerator.Subsection('Description')
    thirdQuartileExample = ReportGenerator.Subsection('Example')
    thirdQUartile.insert_subsections([thirdQuartileDescription,
                             thirdQuartileExample])
    return thirdQuartile


def interquartile_range_section():
    iqr = ReportGenerator.Section('IQR')
    iqrDescription = ReportGenerator.Subsection('Description')
    iqrExample = ReportGenerator.Subsection('Example')
    iqr.insert_subsections([iqrDescription,
                             iqrExample])
    return iqr

def write_chapter():
    chapter = ReportGenerator.Chapter('Quartiles')
    firstQuartile = first_quartile_section()
    secondQuartile = second_quartile_section()
    thirdQuartile = third_quartile_section()
    IQR =  interquartile_range_section()
    chapter.insert_sections([firstQuartile])
    print(chapter)

write_chapter()

"""

"""

def write_chapter(jobDir=None, jobName=None, templatePath='', **kwargs):

    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Sections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections contain content of Figure, Caption, or Paragraph)


    outputPath = os.path.join(jobDir,
                              jobName,
                              'InterQuartileRange.html')

    chapterTitle = 'InterQuartileRange'

    templateVars = {'chapterTitle':
                    chapterTitle,
                'chapterAbstract':
                    ReportGenerator.generate_textstruct(chapterAbstract),
                'chapterSections':
                    ReportGenerator.generate_sentences(chapterSections),
                'heavyBreak':
                    False}
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath,
                                          templateVars=templateVars,
                                          outputPath=outputPath)



def first_quartile_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>1st quartile</var> is the <em>50th Percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 25% of the data exists, and above which 75% of the data exists.'
    ]

    return [p0]

def second_quarile_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>2nd quartile</var> is the <em>50th percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 50% of the data is above and 50% is below'
    ]
    p0['sentence 1'] = [
        'This value is more commonly known as the <var>Median</var>.'
    ]

    return [p0]


def third_quartile_range_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <var>3rd quartile</var> is the <em>75th Percentile</em>.'
    ]
    p0['sentence 1'] = [
        'This is the value below which 75% of the data exists, and above which 25% of the data exists.'
    ]

    return [p0]

def interquartile_range_desccription():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = [
        'The <em>IQR</em> is used to calculate <var>outliers</var>.'
    ]
    p0['sentence 1'] = [
        'Any value that is a distance of <var1.5*IQR</var><em>below the 1st Quartile or above the 3rd Quartile</em> is considered an outlier.'
    ]
    p0['sentence 1'] = [
        'This changes what how the minimum and maximum values are defined'
    ]
    p0['sentence 2'] = [
        'The IQR is calculated as Follows:'
    ]

    return [p0]


def first_quartile_section():
    firstQuartile = ReportGenerator.Section('First Quartile')
    firstQuartileDescription = ReportGenerator.Subsection('Description')
    firstQuartileDescription.insert_content(boxPlot_description())
    firstQuartileExample = ReportGenerator.Subsection('Example')
    firstQuartile.insert_subsections([firstQuartileDescription,
                             firstQuartileExample])
    return firstQuartile


def second_quartile_section():
    secondQuartile = ReportGenerator.Section('Second Quartile')
    secondQuartileDescription = ReportGenerator.Subsection('Description')
    secondQuartileExample = ReportGenerator.Subsection('Example')
    secondQuartile.insert_subsections([secondQuartileDescription,
                               secondQuartileExample])
    return secondQuartile


def third_quartile_section():
    thirdQuartile = ReportGenerator.Section('Third Quartile')
    thirdQuatrileDescription = ReportGenerator.Subsection('Description')
    thirdQuartileExample = ReportGenerator.Subsection('Example')
    thirdQUartile.insert_subsections([thirdQuartileDescription,
                             thirdQuartileExample])
    return thirdQuartile


def interquartile_range_section():
    iqr = ReportGenerator.Section('IQR')
    iqrDescription = ReportGenerator.Subsection('Description')
    iqrExample = ReportGenerator.Subsection('Example')
    iqr.insert_subsections([iqrDescription,
                             iqrExample])
    return iqr

def write_chapter():
    chapter = ReportGenerator.Chapter('Quartiles')
    firstQuartile = first_quartile_section()
    secondQuartile = second_quartile_section()
    thirdQuartile = third_quartile_section()
    IQR =  interquartile_range_section()
    chapter.insert_sections([firstQuartile])
    print(chapter)

write_chapter()

"""

paragraphs = [3, 4, 1]
variations = 1
title = 'Quartiles and Interquartile Range'
abstract = False

# PARAGRAPH 1
h = '''
<h3>
Quartiles
</h3>
'''
s111 = h + '''
<p>
The <var>1st Quartile</var> is the <em>25th Percentile</em>.
This is the value below which 25% of the data exists,
and above which 75% of the data exists.
</p>
'''

s121 = '''
<p>
The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
This is the value below which 
50% of the data is above and 50% is below.
This value is more commonly refered to as the <var>Median</var>.
</p>
'''

s131 = '''
<p>
The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
This is the value below which 75% of the data exists,
and above which 25% of the data exists.
</p>
'''

# PARAGRAPH 1
h = '''
<h3>
Interquartile Range (IQR)
</h3>
'''

s211 = h + '''
<p>
The <em>IQR</em> is used to calculate <var>outliers</var>.
Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or 
above the 3rd Quartile</em> is considered an outlier.
This changes what how the minimum and maximum values are defined.
</p>
'''

img = '''
<image src=''' + iqr + ''' object_oriented="functionString">
'''
s221 = '''
<p>
<em>IQR</em> is calculated as:</br>
</p>
''' + img

img = '''
<image src = ''' + lowB + ''' object_oriented="functionString">
'''
s231 = '''
<p>
The minimum value in the data is defined as:</br>
</p>
''' + img

img = '''
<image src = ''' + uppB + ''' object_oriented="functionString">
'''
s241 = '''
<p>
The maximum value is defined as:</br>
</p>
''' + img

# PARAGRAPH 2
s311 = '''
<p>
If there are no values below the minimum cutoff 
or above the maximum cutoff, 
the minimum and maximum values are simply the 
actual minimum and maximum values in the data.
</p>
'''

paragraphs = [2, 2, 2, 2]
variations = 1
title = 'plot'
abstract = True
# ABSTRACT
a1 = '''
<p>
The following is a brief description of each of the plots presented in this report.
These descriptions indicate how the plots are meant to be interpreted.
</p>
'''

# PARAGRAPH 1
h = '''
<h3>
Box Plot
</h3>
'''

s111 = h + '''
<p>
The box plot0's <em>left</em> whisker shows the <em>lowest</em> value in the distribution. Similarly,
the <em>right</em> whisker shows the <em>highest</em> value in the data.
</p>
'''

s121 = '''
<p>
The <em>left</em> edge of the box is the <em>25th Percentile</em>.
The <em>right</em> edge of the box is the <em>75th Percentile</em>.
Lastly, the <em>line</em> in box is the <em>50th Percentile</em>, or the <em>median</em>.
</p>
'''

# PARAGRAPH 1
h = '''
<h3>
Histogram
</h3>
'''
s211 = h + '''
<p>
The histogram shows the frequency distribution of the data.
</p>
'''

s221 = '''
<p>
Histograms rule!
</p>
'''

# PARAGRAPH 2
h = '''
<h3>
Violin Plot
</h3>
'''
s311 = h + '''
<p>
The violin shows the frequency distribution of the data and has a mini box plot0.
</p>
'''

s321 = '''
<p>
Violin plots are the future!
</p>
'''

# PARAGRAPH 2
h = '''
<h3>
Scatter Plot
</h3>
'''
s411 = h + '''
<p>
The scatter shows the data. These plots show a best fit Regression6
when a significant level of correlation is found.
</p>
'''

s421 = '''
<p>
Scatter plots are beautiful!
</p>
'''

# path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
data = lozoya.signalSyntheticData.df
# data = DataProcessor.read_data(path, None)
# data = data.drop([data.columns[0]], axis=1)
# data = data.set_index(data.columns[0])
dtypes = lozoya.data.get_dtypes(data)
variables = lozoya.plot.StaticPlotGenerator.generate_all_plots(data, dtypes)
chapters, paragraphs = collect_chapters(dtypes)
generate_report_html(chapters, paragraphs)
stats = [0, 0, 0, 0, 0, 0, 0, 0]
generate_report('Hydraulic Data Statistical Analysis', 'Dannenbaum LLC', 'LoPar Technologies LLC', *stats)
env = Environment(loader=FileSystemLoader(''))
template = env.get_template(os.path.join('jinjaTemplate.html'))
path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
# data = lozoya.signalSyntheticData.df
data = DataProcessor.read_data(path, None)
data = data.drop([data.columns[0]], axis=1)
data = data.set_index(data.columns[0])
dtypes = get_dtypes(data)
variables = StaticPlotGenerator.generate_report_plots(data, dtypes)
chapters, paragraphs = collect_chapters(dtypes)
generate_report_html(chapters, paragraphs)
# path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
# data = DataProcessor.read_data(path, None)
# data = data.drop([data.columns[0]], axis=1)
# data = data.set_index(data.columns[0])
dtypes = StaticPlotGenerator.get_dtypes(data)
variables = StaticPlotGenerator.generate_all_plots(data, dtypes)
chapters, paragraphs = collect_chapters(dtypes)
generate_report_html(chapters, paragraphs)
dtypes = get_dtypes(df)
variables = PlotGenerator.generate_report_plots(df, dtypes)
chapters, paragraphs = collect_chapters(dtypes)
ReportGenerator.generate_report_html(chapters, paragraphs)
path_wkthmltopdf = r'C:\Users\frano\AppData\Local\Enthought\Canopy\edm\envs\User\Lib\site-packages\wkhtmltopdf'
config = pdfkit.configuration(wkhtmltopdf=path_wkthmltopdf)
pdfkit.from_file(
    r'C:\Users\frano\PycharmProjects\AutomaticReportGenerator\HTML\Quartiles and Interquartile Range.html',
    'out.pdf'
)
generate_report(os.path.join(gc.server, 'data', 'Iris'))
# path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
data = lozoya.signalSyntheticData.df
# data = DataProcessor.read_data(path, None)
# data = data.drop([data.columns[0]], axis=1)
# data = data.set_index(data.columns[0])
dtypes = StatsCollector.get_dtypes(data)
# variables = StaticPlotGenerator.generate_all_plots(data, dtypes)
# chapters, paragraphs = collect_chapters(dtypes)
# generate_report_html(chapters, paragraphs)
# write_chapter(AnalysisChapter,
#              **{'vars': list(data.columns.values),
#                 'dtypes': dtypes})

# write_chapter(ErrorMetricsChapter)
# path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
data = lozoya.signalSyntheticData.df
# data = DataProcessor.read_data(path, None)
# data = data.drop([data.columns[0]], axis=1)
# data = data.set_index(data.columns[0])
dtypes = StatsCollector.get_dtypes(data)
# variables = StaticPlotGenerator.generate_all_plots(data, dtypes)
# chapters, paragraphs = collect_chapters(dtypes)
# generate_report_html(chapters, paragraphs)
# write_chapter(AnalysisChapter,
#              **{'vars': list(data.columns.values),
#                 'dtypes': dtypes})

# write_chapter(ErrorMetricsChapter)

# path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
# data = lozoya.signalSyntheticData.df
# data = DataProcessor.read_data(path, None)
# data = data.drop([data.columns[0]], axis=1)
# data = data.set_index(data.columns[0])
# dtypes = StatsCollector.get_dtypes(data)
# variables = StaticPlotGenerator.generate_all_plots(data, dtypes)
# chapters, paragraphs = collect_chapters(dtypes)
# generate_report_html(chapters, paragraphs)
# write_chapter(AnalysisChapter,
#              **{'vars': list(data.columns.values),
#                 'dtypes': dtypes})

# write_chapter(ErrorMetricsChapter)
# path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
# data = lozoya.signalSyntheticData.df
# data = DataProcessor.read_data(path, None)
# data = data.drop([data.columns[0]], axis=1)
# data = data.set_index(data.columns[0])
# dtypes = StatsCollector.get_dtypes(data)
# variables = StaticPlotGenerator.generate_all_plots(data, dtypes)
# chapters, paragraphs = collect_chapters(dtypes)
# generate_report_html(chapters, paragraphs)
# write_chapter(AnalysisChapter,
#              **{'vars': list(data.columns.values),
#                 'dtypes': dtypes})

# write_chapter(ErrorMetricsChapter)
"""Automatic Descriptive Statistics report_generator0 Generator"""
# path = r'C:\Users\frano\PycharmProjects\LoPar Technologies\LoParDatasets\CollectedData\WeidnerTrash\CR6Series_Static (med).csv'
# data = lozoya.signalSyntheticData.df
# data = DataProcessor.read_data(path, None)
# data = data.drop([data.columns[0]], axis=1)
# data = data.set_index(data.columns[0])
# dtypes = StatsCollector.get_dtypes(data)
# variables = StaticPlotGenerator.generate_all_plots(data, dtypes)
# chapters, paragraphs = collect_chapters(dtypes)
# generate_report_html(chapters, paragraphs)
# write_chapter(AnalysisChapter,
#              **{'vars': list(data.columns.values),
#                 'dtypes': dtypes})

# write_chapter(ErrorMetricsChapter)
PLOTS_DIR = os.path.join('StaticPlots0')
FORMULAS_DIR = os.path.join('ExpressionStrings1')
beginDoc = str('<!DOCTYPE html>\n<html>\n<link rel="stylesheet" href="ReportCSS4.css">\n<body>')
endDoc = str('</body>\n</html>')
indent = "&emsp;&emsp;&emsp;&emsp;"
subSpaceP = "</br></br></br>"
vars = ['var1', 'var2', 'var3']
analysis = Chapter('Analysis', abstract='This is a test chapter.', sections=vars)
var1 = subsection('var1')
var1['displayTitle'] = False
print(var1)
