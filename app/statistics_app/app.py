import collections
import os
import random
import sys


def bar(var, dtype):
    captions['Bar Plot'] = {}
    preText['Bar Plot'] = {}
    if dtype == 'Categorical':
        captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
        preText['Bar Plot'][var] = random.choice(
            ['''
                                            This is a bar plot0.''', '''
                                                  Welcome to the bar plot0.
                                                  ''']
        )


def box(var, dtype):
    captions['Box Plot'] = {}
    preText['Box Plot'] = {}
    if dtype == 'Numerical':
        captions['Box Plot'][var] = 'Box plot0 of <var>{0}</var>.'.format(var)
        lowerIQR = numStats[var]['Lower IQR']
        upperIQR = numStats[var]['Upper IQR']
        lowerOutliers = numStats[var]['Lower Outliers']
        upperOutliers = numStats[var]['Upper Outliers']
        maximum = numStats[var]['Max']
        minimum = numStats[var]['Min']
        quartile1 = numStats[var]['First Quartile']
        quartile3 = numStats[var]['Third Quartile']
        iqr = numStats[var]['IQR']
        boxText = ''

        # MINIMUM
        boxText += random.choice(
            ['The minimum value found in <var>{0}</var> is <em>{1}</em>. '.format(var, str(minimum))]
        )
        # MAXIMUM
        boxText += random.choice(['The maximum is {}. '.format(str(maximum))])
        # QUARTILES
        boxText += random.choice(
            ['The first quartile is {0} and the third quartile is {1}. '.format(quartile1, quartile3),
             'The first and third quartiles are {0} and {1}, respectively. '.format(quartile1, quartile3)]
        )
        # IQR
        boxText += random.choice(['The interquartile range is therefore {}. '.format(iqr)])
        # LOWER IQR
        boxText += random.choice(['The lower limit of the interquartile range is {}. '.format(lowerIQR)])
        # UPPER IQR
        boxText += random.choice(['and the upper limit is {}. '.format(upperIQR)])
        if upperIQR > maximum:
            boxText += random.choice(
                ['This is above the maximum value in the data. ', 'The maximum value is below this. ']
            )
        else:
            boxText += random.choice(
                ['This is below the maximum value in the data. ', 'The maximum value is above this. ']
            )

        # OUTLIERS
        if len(lowerOutliers) > 0:
            boxText += 'There are {} outliers below the lower bound. '.format(len(lowerOutliers))
        else:
            boxText += 'There are no outliers below the lower bound. '
        if len(upperOutliers) > 0:
            boxText += 'There are {} outliers above the upper bound. '.format(len(upperOutliers))
        else:
            boxText += 'There are no outliers above the upper bound. '

        preText['Box Plot'][var] = boxText


def box(var, dtype):
    captions['Box Plot'] = {}
    preText['Box Plot'] = {}
    if dtype == 'Numerical':
        captions['Box Plot'][var] = 'Box plot0 of <var>{0}</var>.'.format(var)
        lowerIQR = numStats[var]['Lower IQR']
        upperIQR = numStats[var]['Upper IQR']
        lowerOutliers = numStats[var]['Lower Outliers']
        upperOutliers = numStats[var]['Upper Outliers']
        maximum = numStats[var]['Max']
        minimum = numStats[var]['Min']
        quartile1 = numStats[var]['First Quartile']
        quartile3 = numStats[var]['Third Quartile']
        iqr = numStats[var]['IQR']
        boxText = ''

        # MINIMUM
        boxText += random.choice(
            ['The minimum value found in <var>{0}</var> is <em>{1}</em>. '.format(var, str(minimum))]
        )
        # MAXIMUM
        boxText += random.choice(['The maximum is {}. '.format(str(maximum))])
        # QUARTILES
        boxText += random.choice(
            ['The first quartile is {0} and the third quartile is {1}. '.format(quartile1, quartile3),
             'The first and third quartiles are {0} and {1}, respectively. '.format(quartile1, quartile3)]
        )
        # IQR
        boxText += random.choice(['The interquartile range is therefore {}. '.format(iqr)])
        # LOWER IQR
        boxText += random.choice(['The lower limit of the interquartile range is {}. '.format(lowerIQR)])
        # UPPER IQR
        boxText += random.choice(['and the upper limit is {}. '.format(upperIQR)])
        if upperIQR > maximum:
            boxText += random.choice(
                ['This is above the maximum value in the data. ', 'The maximum value is below this. ']
            )
        else:
            boxText += random.choice(
                ['This is below the maximum value in the data. ', 'The maximum value is above this. ']
            )

        # OUTLIERS
        if len(lowerOutliers) > 0:
            boxText += 'There are {} outliers below the lower bound. '.format(len(lowerOutliers))
        else:
            boxText += 'There are no outliers below the lower bound. '
        if len(upperOutliers) > 0:
            boxText += 'There are {} outliers above the upper bound. '.format(len(upperOutliers))
        else:
            boxText += 'There are no outliers above the upper bound. '

        preText['Box Plot'][var] = boxText


def correlation():
    captions['Correlation Plot'] = {}
    preText['Correlation Plot'] = {}
    if 'Numerical' in dtypes:
        captions['Correlation Plot'] = 'Correlation heatmap. '
        correlationText = ''
        correlationText += random.choice(['This is the correlation matrix of the data plotted as a heatmap. ', ])
        preText['Correlation Plot'] = correlationText


def create_section(title):
    section = ReportGenerator.Section(title)
    description = ReportGenerator.Subsection('Description')
    description.insert_content(globals()['{}_description'.format(title.lower())]())
    example = ReportGenerator.Subsection('Example')
    section.insert_subsections([description, example])
    return section


def distribution(var, dtype):
    captions['Distribution Plot'] = {}
    preText['Distribution Plot'] = {}
    if dtype == 'Numerical':
        distributionText = ''
        captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
        distributionText += random.choice(['This plot0 shows the frequency distribution of the data.'])
        distributionText += random.choice(
            ['The rug at the bottom of the plot0 denotes the concentration of the data. ',
             'The rug at the bottom of the plot0 indicates the regions where {} is concentrated. '.format(
                 var
             ), 'The rug at the bottom of the plot0 indicates the concentration of the data. ',
             'The rug at the bottom of the plot0 denotes the regions where {} is concentrated. '.format(
                 var
             ), ]
        )
        preText['Distribution Plot'][var] = distributionText


def distribution_fit(var, dtype):
    captions['Distribution Fit Plot'] = {}
    preText['Distribution Fit Plot'] = {}
    if dtype == 'Numerical':
        keys = list(distFit[var].keys())
        func1 = distFit[var][keys[0]][0]
        # func2 = distFit[var][keys[1]][0]
        error1 = GeneralUtil.round_array(distFit[var][keys[0]][1])
        # error2 = s_round(distFit[var][keys[1]][1])
        captions['Distribution Fit Plot'][var] = 'Distribution fit of <var>{0}</var>.'.format(var)
        distributionFitText = ''
        # Distribution functions
        distributionFitText += random.choice(
            ['Performing a distribution fitting yielded the following best fit function:',
             'Of the distributions that were fit to the data, the following was the best fitting:',
             'This distribution of this data is best approximated by the following function:',
             'The results of curve fitting analysis indicate that the following function best fits the data:']
        )
        distributionFitText += random.choice([r'$$\textbf{' + keys[0] + ':}$$' + '$${}$$'.format(func1)])  # + \
        # r'$$\textbf{' + keys[1] + ':}$$' + \
        # '$${}$$'.format(func2)
        # ])

        # Goodness of fit
        distributionFitText += random.choice(
            ['With <var>{0}</var> having a <em>sum of squared errors</em> of {1}. '.format(keys[0], error1),
             'With <var>{0}</var> having a <em>sum of squared errors</em> of {1}. '.format(keys[0], error1)]
        )

        preText['Distribution Fit Plot'][var] = distributionFitText


def distribution_fit(var, dtype):
    captions['Distribution Fit Plot'] = {}
    preText['Distribution Fit Plot'] = {}
    if dtype == 'Numerical':
        keys = list(distFit[var].keys())
        func1 = distFit[var][keys[0]][0]
        # func2 = distFit[var][keys[1]][0]
        error1 = GeneralUtil.round_array(distFit[var][keys[0]][1])
        # error2 = s_round(distFit[var][keys[1]][1])
        captions['Distribution Fit Plot'][var] = 'Distribution fit of <var>{0}</var>.'.format(var)
        distributionFitText = ''
        # Distribution functions
        distributionFitText += random.choice(
            ['Performing a distribution fitting yielded the following best fit function:',
             'Of the distributions that were fit to the data, the following was the best fitting:',
             'This distribution of this data is best approximated by the following function:',
             'The results of curve fitting analysis indicate that the following function best fits the data:']
        )
        distributionFitText += random.choice([r'$$\textbf{' + keys[0] + ':}$$' + '$${}$$'.format(func1)])  # + \
        # r'$$\textbf{' + keys[1] + ':}$$' + \
        # '$${}$$'.format(func2)
        # ])

        # Goodness of fit
        distributionFitText += random.choice(
            ['With <var>{0}</var> having a <em>sum of squared errors</em> of {1}. '.format(keys[0], error1),
             'With <var>{0}</var> having a <em>sum of squared errors</em> of {1}. '.format(keys[0], error1)]
        )

        preText['Distribution Fit Plot'][var] = distributionFitText


def kurtosis_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice(
        [
            'Kurtosis is the measure of sharpnesss in the peak of the distribution. If it is a positive, negative, or normal curve.',
            'The measure of the shape of the curve. Kurtosis measures if it is flat, normal, or peaked',
            'Kurtosis is the measure of the shape of the distribution. A measure of Kurtosis can either be a 0, a negative, or a positive number',
            'Kurtosis measures how flat or peaked out data distribution is.',
            'Kurtosis refers to the peakedness or flatness of the distribution.']
    )
    p0['sentence 1'] = random.choice(
        ['When the data does not have a peak distribution but resembles a normal bell curve this is called Mesokurtic.',
         'When the curve has the bell curve of a normal distribution then it is a Mesokurtic Distribution',
         'When the calculation of Kurtosis is that of a normal distribution then the distribution is that of a Mesokurtic',
         'When the measure of Kurtosis is a positive distribution it will have a extended peak. This is known as Mesokurtic.',
         'When the distribution has a sharp peak this is a positive kurtosis known as a mesokurtic distribution']
    )
    p0['sentence 1'] = random.choice(
        [
            'A negative kurtosis is a Mesokurtic distribution. This will have closer to what looks like a flat distribution.',
            'When the curve is flatter than a normal bell distribution this is called a Platykurtic distribution',
            'When the calculation of Kurtosis is a negative number this results in a Platykurtic distribution. A peak that is flatter than a normal distribution.',
            'When the measure of Kurtosis has a negative distribution it will have a flatter lower curve this curve is known as Platykurtic.']
    )
    p0['sentence 2'] = random.choice(
        ['A positive kurtosis is called Leptokurtic. In a Leptokurtic distribution the data has a sharp peak',
         'When the curve is taller and skinnier than a normal distribution it has a positive Kurtosis also known as a Letokurtic Distribution',
         'When the calculation of Kurtosis is a positive number this results in a Leptokurtic distribution, or a sharp peak.',
         'When the measure of Kutosis has a regular distribution or a distribution of 0 it will resemble the normal bell ccurve, also known as a Leptokurtic distribution.']
    )
    p0['sentence 3'] = random.choice(['Kurtosis is calculated as follows:', ])
    '''p0['sentence 5'] = random.choice([
        ExpressionGenerator.kurtosis_string()
    ])'''
    return [p0]


def kurtosis_section():
    kurtosis = ReportGenerator.Section('Kurtosis')
    kurtosisDescription = ReportGenerator.Subsection('Description')
    kurtosisExample = ReportGenerator.Subsection('Example')
    kurtosis.insert_subsections([kurtosisDescription, kurtosisExample])
    return kurtosis


def kurtosis_section():
    kurtosis = ReportGenerator.Section('Kurtosis')
    kurtosisDescription = ReportGenerator.Subsection('Description')
    kurtosisExample = ReportGenerator.Subsection('Example')
    kurtosis.insert_subsections([kurtosisDescription, kurtosisExample])
    return kurtosis


def kurtosis_text():
    ('paragraph 0', collections.OrderedDict(
        [('sentence 0', [
            'Kurtosis is the measure of sharpnesss in the peak of the distribution. If it is a positive, negative, or normal curve.',
            'The measure of the shape of the curve. Kurtosis measures if it is flat, normal, or peaked',
            'Kurtosis is the measure of the shape of the distribution. A measure of Kurtosis can either be a 0, a negative, or a positive number',
            'Kurtosis measures how flat or peaked out data distribution is.',
            'Kurtosis refers to the peakedness or flatness of the distribution.']), ('sentence 1', [
            'When the data does not have a peak distribution but resembles a normal bell curve this is called Mesokurtic.',
            'When the curve has the bell curve of a normal distribution then it is a Mesokurtic Distribution',
            'When the calculation of Kurtosis is that of a normal distribution then the distribution is that of a Mesokurtic',
            'When the measure of Kurtosis is a positive distribution it will have a extended peak. This is known as Mesokurtic.',
            'When the distribution has a sharp peak this is a positive kurtosis known as a mesokurtic distribution']), (
             'sentence 1', [
                 'A negative kurtosis is a Mesokurtic distribution. This will have closer to what looks like a flat distribution.',
                 'When the curve is flatter than a normal bell distribution this is called a Platykurtic distribution',
                 'When the calculation of Kurtosis is a negative number this results in a Platykurtic distribution. A peak that is flatter than a normal distribution.',
                 'When the measure of Kurtosis has a negative distribution it will have a flatter lower curve this curve is known as Platykurtic.']),
         ('sentence 2',
          ['A positive kurtosis is called Leptokurtic. In a Leptokurtic distribution the data has a sharp peak',
           'When the curve is taller and skinnier than a normal distribution it has a positive Kurtosis also known as a Letokurtic Distribution',
           'When the calculation of Kurtosis is a positive number this results in a Leptokurtic distribution, or a sharp peak.',
           'When the measure of Kutosis has a regular distribution or a distribution of 0 it will resemble the normal bell ccurve, also known as a Leptokurtic distribution.']),
         ('sentence 3', ['Kurtosis is calculated as follows:', ]),
         ('sentence 5', [ExpressionGenerator.kurtosis_string()])]
    ))


def median_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice(
        [
            'The Median is the midpoint of the data set. It is the point that separates the lower half of your data from the top half.',
            'The midpoint of the data set is known as the Median. The Median is used to represent the center of your data set.',
            'Your data can be split into two sets, a lower half and a top half. The midpoint of the set is known as the Median',
            'The Median is the data point lying at the midpoint of the data set.',
            'The Median is the center value within a data set.', 'The median is the middle value of the data set',
            'The median is the middle most value of data set.']
    )
    p0['sentence 1'] = random.choice(
        [
            'If you want to find the Median of your data set, set all the data points within the data set in numerical order.',
            'In order to get the Median set the data points within the data set in numerical order.',
            'You can get the Median of a data set by setting out all the data points in order.',
            'The Median is found by locating the midpoint within the data set.',
            'Order the values within the data set in numerical order. Look for the center value.',
            'Lay out the data in numerical order.',
            'Lay out the data set from least to greatest and locate the middle value.']
    )

    p0['sentence 1'] = random.choice(
        [
            'If your data set does not have a midpoint due to there being an even number set of data points. Find the mean of the two midpoints.',
            'When the data set has an even number of data points the Mean of the two most center points will be taken and that will be known as the median of the data set.',
            'If the midpoint of your data set cannot be located due to the data having an odd number of data points. Then the two midpoints of the data set will be taken and the Mean of the two points will be found and become the median.',
            'If the data set does not have a center point, find the mean of the two midpoints of the data set. That mean is the Median of the data set.',
            'If the data has an even set of numbers and no center point then calculate the mean of the two center points.',
            'If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points, then the mean of the two mid points will be the median of the data set.']
    )
    '''p0['sentence 2'] = random.choice([
        ExpressionGenerator.median_string()
    ])'''
    return [p0]


def mean_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice(
        ['The Mean of your data set is also known as the average your data set.',
         'The Average of your data set is what is called the Mean or central value of the data set.',
         'The Mean is the average of all data points within the data set and is sometime called the Arithmetic Mean.',
         'One way to find the Central Tendency of your data is to find the Mean (average) of your data set.',
         'The Mean is the Average of your data set.',
         'The average of all the data points within the dataset is called the Mean of the the data set.']
    )
    p0['sentence 1'] = random.choice(
        [
            'The average is calculated by first adding up all the data points within your data set. Take the sum and divide it by the total number of data points within your set.',
            'You must first add up the data points within your data set and then divide by the total number of data points.',
            'The mean is calculated by adding up all the data points and dividing by the total number of data points.',
            'To get the Mean you must add up all the data points. take the sum and divide by the total number of data points.',
            'The Mean is the summation of the data set points divided by the number of dataset points.',
            'To calculate the Average (Mean), add together all the data points within the data set, and then divide the sum by the total number of data points within the data set.',
            'Add all the values together and divide them by the number of data points within the data set.']
    )
    p0['sentence 1'] = random.choice(
        [
            'Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number.',
            'When dealing with a larger data set it may be easier to use a singular value to define it, this value is called the Central Tendency. Calculating the mean is one of the ways to find the Central Tendency.',
            'This gives you a central number to define your data']
    )
    p1 = ReportGenerator.Paragraph('paragraph 1')
    p1['sentence 0'] = random.choice(['The following formula is used for calculating Mean.'])
    p1['sentence 1'] = random.choice(
        ['Mean is calculated as follows:', 'Mean is calculated using the following formula:']
    )
    # p1['sentence 1'] = [ExpressionGenerator.mean_string()]
    return [p0, p1]


def mean_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = ['The Mean of your data set is also known as the average your data set.',
                        'The Average of your data set is what is called the Mean or central value of the data set.',
                        'The Mean is the average of all data points within the data set and is sometime called the Arithmetic Mean.',
                        'One way to find the Central Tendency of your data is to find the Mean (average) of your data set.',
                        'The Mean is the Average of your data set.',
                        'The average of all the data points within the dataset is called the Mean of the the data set.']
    p0['sentence 1'] = [
        'The average is calculated by first adding up all the data points within your data set. Take the sum and divide it by the total number of data points within your set.',
        'You must first add up the data points within your data set and then divide by the total number of data points.',
        'The mean is calculated by adding up all the data points and dividing by the total number of data points.',
        'To get the Mean you must add up all the data points. take the sum and divide by the total number of data points.',
        'The Mean is the summation of the data set points divided by the number of dataset points.',
        'To calculate the Average (Mean), add together all the data points within the data set, and then divide the sum by the total number of data points within the data set.',
        'Add all the values together and divide them by the number of data points within the data set.']
    p0['sentence 1'] = [
        'Often with larger data sets it may be easier to define the data set by a single number, known as the Central Tendency. Calculating the Mean is one way to find this number.',
        'When dealing with a larger data set it may be easier to use a singular value to define it, this value is called the Central Tendency. Calculating the mean is one of the ways to find the Central Tendency.',
        'This gives you a central number to define your data']
    p1 = ReportGenerator.Paragraph('paragraph 1')
    p1['sentence 0'] = ['The following formula is used for calculating Mean.']
    p1['sentence 1'] = ['Mean is calculated as follows:', 'Mean is calculated using the following formula:']
    # p1['sentence 1'] = [ExpressionGenerator.mean_string()]
    return [p0, p1]


def mode_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice(
        ['The Mode of the dataset is the most reccuring data point through out the data set',
         'The Value or frequency to appear within the data set most often is referred to as the Mode',
         'The Mode is the value or data point that appears most often within the data set',
         'When a data set contains a value that occurs more than once it is known as the Mode. This is the Value most likely to be picked from the data set',
         'The value that appears most often is the Mode', 'The value that appears most frequently within the data set']
    ),
    p0['sentence 1'] = random.choice(
        ['One you have found the Median look for the value that is most recurring this the Mode',
         'It is easiest to find the Mode once you have the Median.',
         'After you have found the Median of the data set it ismuch easier to locate which value occurs the most with the data points laid out.',
         'To Calculate the mode align the data points is numerical order and find the data point the occurs the most often.',
         'The Mode is easiest to locate after the median is located. Once you have the data points laid out locate the value that recurrs the most often',
         'Put the data points from least to greatest and count the value that recurrs the most']
    ),
    p0['sentence 1'] = random.choice(
        [
            'A dataset can contain more contain more than one mode. If there are only recurring values the data set is Bimodal.',
            'If the dataset has two recurring values then it is Bimodal',
            'Often a data set may contain more than one recurring value. In such cases the data set is referred to as bimodal',
            'A bimodal data set is a set that contains two Modal Values',
            'A data set containing two modal values is a Bimodal data set',
            'If two values recurr throughout the data an equal amount of times the data set is Bimodal']
    ),
    p0['sentence 2'] = random.choice(
        ['If the dataset has three recurring values then the data set is tri modal',
         'When the data set has three recurring values it is Trimodal',
         'If the data set contains three recurring values. The data set is trimodal',
         'A trimodal data set is a data set that contains three modal values',
         'A data set containing three modal values is a Trimodal',
         'If three values recur throughout the data an equal amount of times the dataset is trimodal']
    ),
    p0['sentence 3'] = random.choice(
        ['If the dataset has more than three recurring values then the data set is multi modal',
         'When the data set has above three recurring values. The data set is multi modal',
         'In larger data sets you may see multiple recurring data sets, In such cases the data set is multi modal',
         'A multi modal set is a dataset that contains multiple recurring values',
         'A data set containing multiple values is Multimodal',
         'If multiple values recur throughout the dataset the set is multimodal']
    )
    return [p0]


def median_section():
    median = ReportGenerator.Section('Median')
    medianDescription = ReportGenerator.Subsection('Description')
    medianExample = ReportGenerator.Subsection('Example')
    median.insert_subsections([medianDescription, medianExample])
    return median


def median_section():
    median = ReportGenerator.Section('Median')
    medianDescription = ReportGenerator.Subsection('Description')
    medianExample = ReportGenerator.Subsection('Example')
    median.insert_subsections([medianDescription, medianExample])
    return median


def median_text():
    ('paragraph 0', collections.OrderedDict(
        [('sentence 0', [
            'The Median is the midpoint of the data set. It is the point that separates the lower half of your data from the top half.',
            'The midpoint of the data set is known as the Median. The Median is used to represent the center of your data set.',
            'Your data can be split into two sets, a lower half and a top half. The midpoint of the set is known as the Median',
            'The Median is the data point lying at the midpoint of the data set.',
            'The Median is the center value within a data set.', 'The median is the middle value of the data set',
            'The median is the middle most value of data set.']), ('sentence 1', [
            'If you want to find the Median of your data set, set all the data points within the data set in numerical order.',
            'In order to get the Median set the data points within the data set in numerical order.',
            'You can get the Median of a data set by setting out all the data points in order.',
            'The Median is found by locating the midpoint within the data set.',
            'Order the values within the data set in numerical order. Look for the center value.',
            'Lay out the data in numerical order.',
            'Lay out the data set from least to greatest and locate the middle value.']), ('sentence 1', [
            'If your data set does not have a midpoint due to there being an even number set of data points. Find the mean of the two midpoints.',
            'When the data set has an even number of data points the Mean of the two most center points will be taken and that will be known as the median of the data set.',
            'If the midpoint of your data set cannot be located due to the data having an odd number of data points. Then the two midpoints of the data set will be taken and the Mean of the two points will be found and become the median.',
            'If the data set does not have a center point, find the mean of the two midpoints of the data set. That mean is the Median of the data set.',
            'If the data has an even set of numbers and no center point then calculate the mean of the two center points.',
            'If the data set has an odd number of data points the midpoint is the median. If the number of data points within the data set equates to an even number of data points, then the mean of the two mid points will be the median of the data set.']),
         ('sentence 2', [ExpressionGenerator.median_string()])]
    ))


def mode_text():
    ('paragraph 0', collections.OrderedDict(
        [('sentence 0', ['The Mode of the dataset is the most reccuring data point through out the data set',
                         'The Value or frequency to appear within the data set most often is referred to as the Mode',
                         'The Mode is the value or data point that appears most often within the data set',
                         'When a data set contains a value that occurs more than once it is known as the Mode. This is the Value most likely to be picked from the data set',
                         'The value that appears most often is the Mode',
                         'The value that appears most frequently within the data set']), ('sentence 1', [
            'One you have found the Median look for the value that is most recurring this the Mode',
            'It is easiest to find the Mode once you have the Median.',
            'After you have found the Median of the data set it ismuch easier to locate which value occurs the most with the data points laid out.',
            'To Calculate the mode align the data points is numerical order and find the data point the occurs the most often.',
            'The Mode is easiest to locate after the median is located. Once you have the data points laid out locate the value that recurrs the most often',
            'Put the data points from least to greatest and count the value that recurrs the most', ]), ('sentence 1', [
            'A dataset can contain more contain more than one mode. If there are only recurring values the data set is Bimodal.',
            'If the dataset has two recurring values then it is Bimodal',
            'Often a data set may contain more than one recurring value. In such cases the data set is referred to as bimodal',
            'A bimodal data set is a set that contains two Modal Values',
            'A data set containing two modal values is a Bimodal data set',
            'If two values recurr throughout the data an equal amount of times the data set is Bimodal']), (
             'sentence 2', ['If the dataset has three recurring values then the data set is tri modal',
                            'When the data set has three recurring values it is Trimodal',
                            'If the data set contains three recurring values. The data set is trimodal',
                            'A trimodal data set is a data set that contains three modal values',
                            'A data set containing three modal values is a Trimodal',
                            'If three values recur throughout the data an equal amount of times the dataset is trimodal']),
         ('sentence 3', ['If the dataset has more than three recurring values then the data set is multi modal',
                         'When the data set has above three recurring values. The data set is multi modal',
                         'In larger data sets you may see multiple recurring data sets, In such cases the data set is multi modal',
                         'A multi modal set is a dataset that contains multiple recurring values',
                         'A data set containing multiple values is Multimodal',
                         'If multiple values recur throughout the dataset the set is multimodal'])]
    ))


def mean_section():
    mean = ReportGenerator.Section('Mean')
    meanDescription = ReportGenerator.Subsection('Description')
    meanDescription.insert_content(mean_description())
    meanExample = ReportGenerator.Subsection('Example')
    mean.insert_subsections([meanDescription, meanExample])
    return mean


def mode_section():
    mode = ReportGenerator.Section('Mode')
    modeDescription = ReportGenerator.Subsection('Description')
    modeExample = ReportGenerator.Subsection('Example')
    mode.insert_subsections([modeDescription, modeExample])
    return mode


def skew_section():
    skew = ReportGenerator.Section('Skew')
    skewDescription = ReportGenerator.Subsection('Description')
    skewExample = ReportGenerator.Subsection('Example')
    skew.insert_subsections([skewDescription, skewExample])
    return skew


def scatter(var, dtype):
    captions['Scatter Plot'] = {}
    preText['Scatter Plot'] = {}
    if dtype == 'Numerical':
        fit = GeneralUtil.round_array(regression[var][0])
        errors = regression[var][1]
        mse = GeneralUtil.round_array(errors['MSE'])
        R2 = GeneralUtil.round_array(errors['R Squared'])
        rmse = GeneralUtil.round_array(errors['RMSE'])
        mae = GeneralUtil.round_array(errors['MAE'])
        sor = GeneralUtil.round_array(errors['Sum of Residuals'])
        chi2 = GeneralUtil.round_array(errors['Chi Squared'])
        reducedChi2 = GeneralUtil.round_array(errors['Reduced Chi Squared'])
        standardError = GeneralUtil.round_array(errors['Standard Error'])
        captions['Scatter Plot'][var] = 'Scatter plot0 and Regression6 fit of <var>{0}</var>.'.format(var)
        scatterText = ''
        scatterText += random.choice(
            ['Using curve fitting procedures, '
             '<var>{0}</var> was found to be best described by $${1}$$'.format(var, fit),

             'Through a Regression6 analysis, the function'
             '$${1}$$ was found to best fit <var>{0}</var>. '.format(var, fit),

             'Upon performing a Regression6 analysis, the data was best fit by '
             '$${0}$$ '.format(fit),

             'The data is best approximated by the following function: '
             '$${0}$$ '.format(fit),

             '<var>{0}</var> is best approximated by the following function: '
             '$${1}$$ '.format(var, fit),

             ]
        )
        scatterText += random.choice(['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)])
        scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
        scatterText += random.choice(['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)])
        scatterText += random.choice(['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)])
        scatterText += random.choice(
            ['The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
             'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
        )
        scatterText += random.choice(
            ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
             'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(standardError, sor)]
        )

        preText['Scatter Plot'][var] = scatterText


def scatter(var, dtype):
    captions['Scatter Plot'] = {}
    preText['Scatter Plot'] = {}
    if dtype == 'Numerical':
        fit = GeneralUtil.round_array(regression[var][0])
        errors = regression[var][1]
        mse = GeneralUtil.round_array(errors['MSE'])
        R2 = GeneralUtil.round_array(errors['R Squared'])
        rmse = GeneralUtil.round_array(errors['RMSE'])
        mae = GeneralUtil.round_array(errors['MAE'])
        sor = GeneralUtil.round_array(errors['Sum of Residuals'])
        chi2 = GeneralUtil.round_array(errors['Chi Squared'])
        reducedChi2 = GeneralUtil.round_array(errors['Reduced Chi Squared'])
        standardError = GeneralUtil.round_array(errors['Standard Error'])
        captions['Scatter Plot'][var] = 'Scatter plot0 and Regression6 fit of <var>{0}</var>.'.format(var)
        scatterText = ''
        scatterText += random.choice(
            ['Using curve fitting procedures, '
             '<var>{0}</var> was found to be best described by $${1}$$'.format(var, fit),

             'Through a Regression6 analysis, the function'
             '$${1}$$ was found to best fit <var>{0}</var>. '.format(var, fit),

             'Upon performing a Regression6 analysis, the data was best fit by '
             '$${0}$$ '.format(fit),

             'The data is best approximated by the following function: '
             '$${0}$$ '.format(fit),

             '<var>{0}</var> is best approximated by the following function: '
             '$${1}$$ '.format(var, fit),

             ]
        )
        scatterText += random.choice(['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)])
        scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
        scatterText += random.choice(['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)])
        scatterText += random.choice(['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)])
        scatterText += random.choice(
            ['The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
             'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
        )
        scatterText += random.choice(
            ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
             'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(standardError, sor)]
        )

        preText['Scatter Plot'][var] = scatterText


def skew_description():
    p0 = ReportGenerator.Paragraph('paragraph 0')
    p0['sentence 0'] = random.choice(
        [
            'The Skew of a data set is the measure of the lack of symmetry within the data. The Skewness of the data can be either positive, negative, or undefined',
            'Skewness describes the symmetry of a distribution. Or lack there of symmetry. If the data has a tail end distribution then the data is skewed.',
            'The Distribution of a data set is the Skewness of the data set. This defines whether the data is symmetrical or not',
            'The skew defines the distribution of the data set.',
            'Skewness is when the curve of the data is distorted to either the left or the right']
    )
    p0['sentence 1'] = random.choice(
        [
            'If the data is skewed to the right this is known as a positive distribution. In a positive distribution the mean usually greater than the median, both are to the right of the Mode.',
            'If the tail of the data extends to the right then the data is positively skewed. In a positively skewed distribution the Mean is usually to the right of the median and both are to the right of the Mode.',
            'When the tail of the data lays to the right of the curve. The data has a positive skew. In a positive skew the Mean is greater than the median and both are to the right of the mode.',
            'If the right side of the data has a tail then the data is positively skewed. The mean and the median lie to the right of the mode.',
            'A curve aligned to the right extending a tail to the left is a negatively skewed curve.']
    )
    p0['sentence 1'] = random.choice(
        [
            'If the data is skewed to the left this is known as a negative distribution. In a negative distribution the Mean is less than the median and both are less than the mode.',
            'If the tail of the data extends to the left then the data is negatively skewed. In a negative distribution the Mean is to the left of the median and both are to the left of the mode.',
            'When the tail of the data lays to the left of the curve the mean is less than the median and both are to the left of the curve.',
            'If the left side of the data has a tail end. The data is negatively skewed. The mean is less than the median and the Mode.',
            'A curve aligned to the left extending a tail to the right is a positively skewed curve.']
    )
    p0['sentence 2'] = random.choice(
        ['The distribution of the data is symmetrical when the data lacks skewness.',
         'When the data does not contain a tail it is a bell curve. In a bell curve the data is symmetrical.',
         'If there is no extending tail in the data the data is a Bell Curve. A symmetrical data set',
         'A data set is symmetrical if it looks the same on the left and the right side.',
         'A curve in the center of the data is a bell curve, a symmetrical curve.']
    )
    p0['sentence 3'] = random.choice(['The Skew is calculated as follows:', ])
    '''p0['sentence 5'] = random.choice([
        ExpressionGenerator.skew_string()
    ])'''
    return [p0]


def skew_text():
    ('paragraph 0', collections.OrderedDict(
        [('sentence 0', [
            'The Skew of a data set is the measure of the lack of symmetry within the data. The Skewness of the data can be either positive, negative, or undefined',
            'Skewness describes the symmetry of a distribution. Or lack there of symmetry. If the data has a tail end distribution then the data is skewed.',
            'The Distribution of a data set is the Skewness of the data set. This defines whether the data is symmetrical or not',
            'The skew defines the distribution of the data set.',
            'Skewness is when the curve of the data is distorted to either the left or the right']), ('sentence 1', [
            'If the data is skewed to the right this is known as a positive distribution. In a positive distribution the mean usually greater than the median, both are to the right of the Mode.',
            'If the tail of the data extends to the right then the data is positively skewed. In a positively skewed distribution the Mean is usually to the right of the median and both are to the right of the Mode.',
            'When the tail of the data lays to the right of the curve. The data has a positive skew. In a positive skew the Mean is greater than the median and both are to the right of the mode.',
            'If the right side of the data has a tail then the data is positively skewed. The mean and the median lie to the right of the mode.',
            'A curve aligned to the right extending a tail to the left is a negatively skewed curve.']), ('sentence 1',
                                                                                                          [
                                                                                                              'If the data is skewed to the left this is known as a negative distribution. In a negative distribution the Mean is less than the median and both are less than the mode.',
                                                                                                              'If the tail of the data extends to the left then the data is negatively skewed. In a negative distribution the Mean is to the left of the median and both are to the left of the mode.',
                                                                                                              'When the tail of the data lays to the left of the curve the mean is less than the median and both are to the left of the curve.',
                                                                                                              'If the left side of the data has a tail end. The data is negatively skewed. The mean is less than the median and the Mode.',
                                                                                                              'A curve aligned to the left extending a tail to the right is a positively skewed curve.']),
         ('sentence 2', ['The distribution of the data is symmetrical when the data lacks skewness.',
                         'When the data does not contain a tail it is a bell curve. In a bell curve the data is symmetrical.',
                         'If there is no extending tail in the data the data is a Bell Curve. A symmetrical data set',
                         'A data set is symmetrical if it looks the same on the left and the right side.',
                         'A curve in the center of the data is a bell curve, a symmetrical curve.']),
         ('sentence 3', ['The Skew is calculated as follows:', ]), ('sentence 5', [ExpressionGenerator.skew_string()])]
    ))


def violin(var, dtype):
    captions['Violin Plot'] = {}
    preText['Violin Plot'] = {}
    if dtype == 'Numerical':
        captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
        mean = GeneralUtil.round_array(numStats[var]['Mean'])
        median = GeneralUtil.round_array(numStats[var]['Median'])
        mode = GeneralUtil.round_array(numStats[var]['Mode'])
        std = GeneralUtil.round_array(numStats[var]['Standard Deviation'])
        variance = GeneralUtil.round_array(numStats[var]['Variance'])
        skew = GeneralUtil.round_array(numStats[var]['Skew'])
        kurtosis = GeneralUtil.round_array(numStats[var]['Kurtosis'])
        x = GeneralUtil.round_array(mean - median)
        c0 = GeneralUtil.round_array(mean / median)
        c1 = GeneralUtil.round_array((abs(x) / mean) * 100)

        # MEAN ----------------------------------------------
        meanSentence = random.choice(
            ['The distribution of <var>' + var + '</var> has a <em>mean</em> of ' + str(mean),

             '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(
                 mean
             ) + '</em>']
        ) + end

        # MEDIAN --------------------------------------------
        medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
        if x > 0:
            '''Mean is greater than median'''
            meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                   'times, or {1} % greater than ' \
                                   'the <em>median</em>. '.format(str(c0), str(c1))
        else:
            '''Median is greater than mean'''
            meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                   'times, or {1} % smaller than ' \
                                   'the <em>median</em>. '.format(str(GeneralUtil.round_array(1 / c0)), str(c1))

        # MODE ----------------------------------------------
        modeSentence = ''

        # STANDARD DEVIATION --------------------------------
        stdSentence = 'This distribution has a <em>standard deviation</em> ' \
                      'of {}. '.format(str(std))

        # VARIANCE ------------------------------------------
        varianceSentence = 'Therefore, the <em>variance</em> is ' \
                           '{}. '.format(str(variance))

        # SKEW ----------------------------------------------
        skewText = random.choice(
            ['The <em>skew</em> of the distribution is ' + str(
                skew
            ) + ', which indicates that the data is concentrated ']
        )

        if abs(skew) <= 0.5:
            '''fairly symmetrical'''
            skewText += random.choice(['slightly'])
        elif abs(skew) <= 1:
            '''moderately skew'''
            skewText += random.choice(['moderately'])
        else:
            '''highly skewed'''
            skewText += random.choice(['highly'])

        if skew > 0:
            skewText += ' below the mean with a longer tail above the mean. '
        elif skew < 0:
            skewText += ' above the mean with a longer tail below the mean. '
        else:
            skewText += ' centered with equally long tails above and below the mean.' \
                        'Thus this distribution is perfectly symmetrical'

        # KURTOSIS ------------------------------------------
        kurtosisText = random.choice(['The <em>kurtosis</em> of ' + str(kurtosis) + ' means that the data is '])
        if kurtosis < 0:
            kurtosisText += ' a very spread out' \
                            ' <em>platykurtic</em> distribution. Therefore, the tails of the distribution' \
                            ' are relatively long, slender, and more prone to contain outliers. '
        elif kurtosis > 0:
            kurtosisText += ' a tightly concentrated' \
                            ' <em>leptokurtic</em> distribution. Therefore, the tails of the distribution' \
                            ' are relatively short, broad, and less prone to contain outliers. '
        else:
            kurtosisText += ' a moderately spread out' \
                            ' <em>mesokurtic</em> distribution. '

        violinText = meanSentence + ' ' + medianSentence + ' ' + meanMedianComparison + ' ' + stdSentence + ' ' + varianceSentence + ' ' + skewText + ' ' + kurtosisText

        preText['Violin Plot'][var] = violinText


def violin(var, dtype):
    captions['Violin Plot'] = {}
    preText['Violin Plot'] = {}
    if dtype == 'Numerical':
        captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
        mean = GeneralUtil.round_array(numStats[var]['Mean'])
        median = GeneralUtil.round_array(numStats[var]['Median'])
        mode = GeneralUtil.round_array(numStats[var]['Mode'])
        std = GeneralUtil.round_array(numStats[var]['Standard Deviation'])
        variance = GeneralUtil.round_array(numStats[var]['Variance'])
        skew = GeneralUtil.round_array(numStats[var]['Skew'])
        kurtosis = GeneralUtil.round_array(numStats[var]['Kurtosis'])
        x = GeneralUtil.round_array(mean - median)
        c0 = GeneralUtil.round_array(mean / median)
        c1 = GeneralUtil.round_array((abs(x) / mean) * 100)

        # MEAN ----------------------------------------------
        meanSentence = random.choice(
            ['The distribution of <var>' + var + '</var> has a <em>mean</em> of ' + str(mean),

             '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(
                 mean
             ) + '</em>']
        ) + end

        # MEDIAN --------------------------------------------
        medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
        if x > 0:
            '''Mean is greater than median'''
            meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                   'times, or {1} % greater than ' \
                                   'the <em>median</em>. '.format(str(c0), str(c1))
        else:
            '''Median is greater than mean'''
            meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                   'times, or {1} % smaller than ' \
                                   'the <em>median</em>. '.format(str(GeneralUtil.round_array(1 / c0)), str(c1))

        # MODE ----------------------------------------------
        modeSentence = ''

        # STANDARD DEVIATION --------------------------------
        stdSentence = 'This distribution has a <em>standard deviation</em> ' \
                      'of {}. '.format(str(std))

        # VARIANCE ------------------------------------------
        varianceSentence = 'Therefore, the <em>variance</em> is ' \
                           '{}. '.format(str(variance))

        # SKEW ----------------------------------------------
        skewText = random.choice(
            ['The <em>skew</em> of the distribution is ' + str(
                skew
            ) + ', which indicates that the data is concentrated ']
        )

        if abs(skew) <= 0.5:
            '''fairly symmetrical'''
            skewText += random.choice(['slightly'])
        elif abs(skew) <= 1:
            '''moderately skew'''
            skewText += random.choice(['moderately'])
        else:
            '''highly skewed'''
            skewText += random.choice(['highly'])

        if skew > 0:
            skewText += ' below the mean with a longer tail above the mean. '
        elif skew < 0:
            skewText += ' above the mean with a longer tail below the mean. '
        else:
            skewText += ' centered with equally long tails above and below the mean.' \
                        'Thus this distribution is perfectly symmetrical'

        # KURTOSIS ------------------------------------------
        kurtosisText = random.choice(['The <em>kurtosis</em> of ' + str(kurtosis) + ' means that the data is '])
        if kurtosis < 0:
            kurtosisText += ' a very spread out' \
                            ' <em>platykurtic</em> distribution. Therefore, the tails of the distribution' \
                            ' are relatively long, slender, and more prone to contain outliers. '
        elif kurtosis > 0:
            kurtosisText += ' a tightly concentrated' \
                            ' <em>leptokurtic</em> distribution. Therefore, the tails of the distribution' \
                            ' are relatively short, broad, and less prone to contain outliers. '
        else:
            kurtosisText += ' a moderately spread out' \
                            ' <em>mesokurtic</em> distribution. '

        violinText = meanSentence + ' ' + medianSentence + ' ' + meanMedianComparison + ' ' + stdSentence + ' ' + varianceSentence + ' ' + skewText + ' ' + kurtosisText

        preText['Violin Plot'][var] = violinText
