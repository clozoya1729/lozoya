'''
import calendar
import shutil
import PyPDF2 as pdf

import pdfkit
from jinja2 import Environment, FileSystemLoader

import configuration as gc
import data as util
import lozoya.signal
from configuration import beginDoc, endDoc
import lozoya.data
import lozoya.plot


# TODO https://stackoverflow.com/questions/23359083/how-to-convert-webpage-into-pdf-by-using-python


class BaseSection():
    def __init__(self, name, abstract='', breakAfter=True, displayTitle=True):
        self.name = name
        self.abstract = ''
        self.breakAfter = ''
        self.displayTitle = displayTitle


def subsection(title, displayTitle=True, hide=False):
    _ss = [('breakAfter', True), ('column', 'center'), ('content', []), ('displayTitle', displayTitle), ('hide', hide),
           ('title', title)]
    ss = OrderedDict(_ss)
    return ss


def get_dtypes(df):
    """
    This is gonna have ai to find dtype
    :param df:
    :return:
    """
    return ['Numerical']


def render_HTML(templatePath, templateVars, outputPath):
    templateDir = os.path.dirname(templatePath)
    template = os.path.basename(os.path.normpath(templatePath))
    env = Environment(loader=FileSystemLoader(templateDir))
    template = env.get_template(template)
    htmlOut = template.render(templateVars)
    with open(outputPath, 'w') as f:
        f.write(htmlOut)


def generate_textstruct(textStruct):
    """
    textStruct: list of str
    return:
    """
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def Subsection(title, breakAfter=True, column='center', displayTitle=True, hide=False):
    _ss = [('title', title), ('displayTitle', displayTitle), ('content', [])]
    ss = _Subsection(_ss)
    return ss


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


def _collect(path):
    soup = bs(open(path), features='html.parser')
    headers = soup.find_all(re.compile('^h[1-6]$'))
    return [str(header) for header in headers]


def collect_headers(paths):
    headers = []
    for path in paths:
        headers.extend(_collect(path))
    return headers


def _parse(header):
    size = header[1:3]
    _id = re.findall(r'id=".*"', header)[0]
    id = _id[4:-1].replace(' ', '')
    _name = re.findall(r'>.*<', header)[0]
    name = _name[1:-1]
    _dict = OrderedDict([('size', size), ('name', name), ('id', id), ('subheaders', [])])
    return _dict


def parse_headers(headers):
    """
    header: string, A string following the format of <hi id="id">name</hi>
    where i is a number from 1 to 6
    return:
    """
    headersDict = {}
    h2 = []
    h3 = []
    h4 = []
    for header in headers:
        _dict = _parse(header)
        if _dict['size'] == 'h2':
            h2.append(_dict['id'])
            headersDict[_dict['id']] = _dict
        elif _dict['size'] == 'h3':
            h3.append(_dict['id'])
            headersDict[h2[-1]]['subheaders'].append(_dict)
        elif _dict['size'] == 'h4':
            h4.append(_dict['id'])
            headersDict[h2[-1]]['subheaders'][-1]['subheaders'].append(_dict)
    m = OrderedDict([(header, headersDict[header]) for header in h2])
    return m


def collect_headers(paths):
    headers = []
    for path in paths:
        headers.extend(_collect(path))
    return headers


def parse_headers(path):
    """
    header: string, A string following the format of <h2 id="id">name</h2>
    return:
    """
    headers = []
    headers.extend(_collect(path))
    headersDict = {}
    h2 = []
    h3 = []
    h4 = []
    for header in headers:
        _dict = _parse(header)
        if _dict['size'] == 'h2':
            h2.append(_dict['id'])
            headersDict[_dict['id']] = _dict
        elif _dict['size'] == 'h3':
            h3.append(_dict['id'])
            headersDict[h2[-1]]['subheaders'].append(_dict)
        elif _dict['size'] == 'h4':
            h4.append(_dict['id'])
            headersDict[h2[-1]]['subheaders'][-1]['subheaders'].append(_dict)
    m = OrderedDict([(header, headersDict[header]) for header in h2])
    return m


def navigator(**kwargs):
    headersDict = parse_headers(kwargs['analysisPath'])
    templateVars = {'headersDict': headersDict}
    render_HTML(templatePath=gc.navTemplatePath, templateVars=templateVars, outputPath=kwargs['navPath'])


def analysis(data=None, **kwargs):
    """
    vars: list of strings
    dtypes: list of strings
    return:
    """
    imagesPath = os.path.join(kwargs['jobDir'], gc.jobImgPath)
    vars, numStats, catStats, distFit, regression = plot_api.generate_all_plots(data, kwargs['dtypes'], path=imagesPath)
    plots = OrderedDict(
        [(var, [('Scatter Plots', 'Regression Error Tables'),
                ('Distribution Plots', 'Central Tendencies Tables'),
                ('Distribution Fit Plots', 'Distribution Error Tables'), ('Box Plots', 'IQR Tables'),
                ('Violin Plots', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar Plots', '')])
         for var, dtype in zip(vars, kwargs['dtypes'])]
    )
    captions = {}
    text = {}

    captions['Bar Plot'] = {}
    text['Bar Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Categorical':
            captions['Bar Plot'][var] = 'Bar Plot of <var>{}</var>.'.format(var)
            text['Bar Plot'][var] = ''

    captions['Box Plot'] = {}
    text['Box Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            captions['Box Plot'][var] = 'Box plot of <var>{}</var>.'.format(var)
            lowerIQR = numStats[var]['Lower IQR']
            upperIQR = numStats[var]['Upper IQR']
            lowerOutliers = numStats[var]['Lower Outliers']
            upperOutliers = numStats[var]['Upper Outliers']
            maximum = numStats[var]['Max']
            minimum = numStats[var]['Min']
            quartile1 = numStats[var]['First Quartile']
            quartile3 = numStats[var]['Third Quartile']
            iqr = numStats[var]['IQR']
            boxText = 'The minimum value found in <var>{}</var> is <em>{}</em>. '.format(var, str(minimum))
            boxText += 'The maximum is {}. '.format(str(maximum))
            boxText += 'The first quartile is {} and the third quartile is {}. '.format(quartile1, quartile3)
            boxText += 'The interquartile range is therefore {}. '.format(iqr)
            boxText += 'The lower limit of the interquartile range is {}. '.format(lowerIQR)
            boxText += 'and the upper limit is {}. '.format(upperIQR)
            if upperIQR > maximum:
                boxText += 'This is above the maximum value in the data. '
            else:
                boxText += 'This is below the maximum value in the data. '
            if len(lowerOutliers) > 0:
                boxText += 'There are {} outliers below the lower bound. '.format(len(lowerOutliers))
            else:
                boxText += 'There are no outliers below the lower bound. '
            if len(upperOutliers) > 0:
                boxText += 'There are {} outliers above the upper bound. '.format(len(upperOutliers))
            else:
                boxText += 'There are no outliers above the upper bound. '
            text['Box Plot'][var] = boxText

    captions['Distribution Plot'] = {}
    text['Distribution Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            distributionText = ''
            captions['Distribution Plot'][var] = 'Histogram of <var>{}</var>.'.format(var)
            text['Distribution Plot'][var] = distributionText

    captions['Distribution Fit Plot'] = {}
    text['Distribution Fit Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            keys = list(distFit[var].keys())
            func1 = distFit[var][keys[0]][0]
            # func2 = distFit[var][keys[1]][0]
            error1 = util.round_array(distFit[var][keys[0]][1])
            # error2 = s_round(distFit[var][keys[1]][1])
            captions['Distribution Fit Plot'][var] = 'Distribution fit of <var>{0}</var>.'.format(var)
            # Distribution functions
            distributionFitText = r'$$\textbf{' + keys[0] + ':}$$' + '$${}$$'.format(
                func1
            )  # + r'$$\textbf{' + keys[1] + ':}$$' + '$${}$$'.format(func2) ])
            text['Distribution Fit Plot'][var] = distributionFitText

    captions['Scatter Plot'] = {}
    text['Scatter Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            fit = util.round_array(regression[var][0])
            errors = regression[var][1]
            captions['Scatter Plot'][var] = 'Scatter plot and regression fit of <var>{}</var>.'.format(var)
            scatterText = '<var>{}</var> is best approximated by the following function: $${}$$ '.format(var, fit)
            text['Scatter Plot'][var] = scatterText

    captions['Violin Plot'] = {}
    text['Violin Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            captions['Violin Plot'][var] = 'Violin plot of <var>{0}</var>.'.format(var)
            mean = util.round_array(numStats[var]['Mean'])
            median = util.round_array(numStats[var]['Median'])
            std = util.round_array(numStats[var]['Standard Deviation'])
            variance = util.round_array(numStats[var]['Variance'])
            skew = util.round_array(numStats[var]['Skew'])
            kurtosis = util.round_array(numStats[var]['Kurtosis'])
            x = util.round_array(mean - median)
            c0 = util.round_array(mean / median)
            c1 = util.round_array((abs(x) / mean) * 100)
            if x > 0:  # mean > median
                meanMedianCompare = 'The <em>mean</em> is {} times, or {}% greater than the <em>median</em>. '.format(
                    str(c0), str(c1)
                )
            else:  # mean < median
                meanMedianCompare = 'The <em>mean</em> is {} times, or {}% smaller than the <em>median</em>. '.format(
                    str(util.round_array(1 / c0)), str(c1)
                )
            # STANDARD DEVIATION --------------------------------
            stdText = 'This distribution has a <em>standard deviation</em> of {}. '.format(str(std))
            # VARIANCE ------------------------------------------
            varianceText = 'Therefore, the <em>variance</em> is {}. '.format(str(variance))
            # SKEW ----------------------------------------------
            skewText = 'The distribution <em>skew</em>  is {}, which indicates that the data is concentrated '.format(
                skew
            )
            if abs(skew) <= 0.5:
                skewText += 'slightly'
            elif abs(skew) <= 1:
                skewText += 'moderately'
            else:
                skewText += 'highly'
            if skew > 0:
                skewText += ' below the mean with a longer tail above the mean. '
            elif skew < 0:
                skewText += ' above the mean with a longer tail below the mean. '
            else:
                skewText += ' centered with symmetrical tails.'
            # KURTOSIS ------------------------------------------
            kurtosisText = 'The <em>kurtosis</em> of {} means that the data is '.format(kurtosis)
            if kurtosis < 0:
                kurtosisText += ' a very spread out<em>platykurtic</em> distribution. The tails of the distribution' \
                                ' are relatively long, slender, and more prone to contain outliers. '
            elif kurtosis > 0:
                kurtosisText += ' a tightly concentrated<em>leptokurtic</em> distribution. The tails of the' \
                                '  distribution are relatively short, broad, and less prone to contain outliers. '
            else:
                kurtosisText += ' a moderately spread out <em>mesokurtic</em> distribution. '
            violinText = '{} {} {} {} {}'.format(meanMedianCompare, stdText, varianceText, skewText, kurtosisText)
            text['Violin Plot'][var] = violinText

    captions['Correlation Plot'] = {}
    if 'Numerical' in kwargs['dtypes']:
        captions['Correlation Plot'] = 'Correlation heatmap. '
        correlationText = 'This is the correlation matrix of the data plotted as a heatmap. '
        text['Correlation Plot'] = correlationText

    chapterSections = OrderedDict(
        [  # SECTION ----------------------
            (var, {  # SUBSECTIONS --------------
                'subSections': OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'figures': {
                                'Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ): {
                                    'image':   os.path.join(jobImgPath, plot[0].replace(' ', '')[:-5], var + '.png'),
                                    'caption': '<var>Figure {}:</var> {}'.format(
                                        (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]),
                                        captions[plot[0][:-1]][var]
                                    )
                                },
                                'Figure {}'.format(
                                    (i + 2) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ): {
                                    'image':   os.path.join(jobImgPath, plot[1].replace(' ', '')[:-6], var + '.png'),
                                    'caption': '<var>Table {0}:</var> {1}'.format(
                                        (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]),
                                        captions[plot[0][:-1]][var]
                                    )
                                },
                            }, 'text': text[plot[0][:-1]][var],
                        }) for i, plot in
                        enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )
    if 'Numerical' in kwargs['dtypes']:
        chapterSections.update(
            {
                'Correlation': {  # SUBSECTIONS --------------
                    'subSections': OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'figures': {
                                    'Figure -1': {
                                        'image':   os.path.join(jobImgPath, 'Heatmap', 'Correlation.png'),
                                        'caption': '<var>Figure {}:</var> {}'.format(
                                            1 + sum([len(plots[v]) for v in vars]),
                                            captions['Correlation Plot']
                                        )
                                    },
                                },
                                'text':    text['Correlation Plot'],
                            })]
                    )
                }
            }
        )
    chapterSections.move_to_end('Correlation', last=True)
    templateVars = {'chapterTitle': 'Analysis', 'chapterSections': chapterSections}
    render_HTML(templatePath=gc.analysisTemplatePath, templateVars=templateVars, outputPath=kwargs['analysisPath'])


def merge_chapters(resultPath='', paths='', export=True, **kwargs):
    soups = [bs(open(path), features='html.parser') for path in paths]
    content = merge(soups)
    if export:
        with open(resultPath, 'w') as file:
            file.write(content)


def make_dirs(kwargs):
    os.mkdir(kwargs['plotsDir'])
    s = ['bar', 'box', 'centraltendencies', 'dispersion', 'distribution', 'distributionerror', 'distributionfit',
         'heatmap', 'iqr', 'regressionerror', 'scatter', 'statscompare', 'table', 'violin']
    for d in s:
        os.mkdir(os.path.join(kwargs['plotsDir'], d))


def generate_report_html(chapters=None, paragraphs=None):
    def write_chapter(chapter, paragraphs):
        with open(os.path.join('../../../app/data/report_generator/html/HTML4', chapter.title + '.html'), 'w') as f:
            f.write(beginDoc)
            for p in (paragraphs[chapter]):
                for para in p:
                    f.write(para)
            f.write(endDoc)

    paragraph_function(chapters, paragraphs, write_chapter)


def paragraph_function(chapters, paragraphs, write_chapter):
    for chapter in chapters:
        for paragraph in range(1, len(chapter.paragraphs) + 1):
            para = []
            if paragraph == 1:
                para.append(headerize(chapter.title, 2))
                if chapter.abstract:
                    r = np.random.randint(1, chapter.variations + 1)
                    a = getattr(chapter, 'a' + str(r))
                    para.append(a)
            for sentence in range(1, chapter.paragraphs[paragraph - 1] + 1):
                r = np.random.randint(1, chapter.variations + 1)
                s = 's' + str(paragraph) + str(sentence) + str(r)
                sentence = getattr(chapter, s)
                para.append(sentence)
            paragraphs[chapter].append(para)
        write_chapter(chapter, paragraphs)


def collect_chapters(dtypes):
    """
    dtypes: list of str
    return:
    """
    chapters = [Plots]
    if 'Categorical' in dtypes:
        pass
    if 'Numerical' in dtypes:
        chapters.append(ErrorMetrics)
        chapters.append(Quartiles_IQR)
        chapters.append(DescriptiveStatistics)
    paragraphs = {chapter: [] for chapter in chapters}
    return chapters, paragraphs


def generate_report_html(chapters=None, paragraphs=None):
    def write_chapter(chapter, paragraphs):
        with open(os.path.join('../../../app/data/report_generator/html/HTML3', chapter.title + '.html'), 'w') as f:
            f.write(beginDoc)
            for p in (paragraphs[chapter]):
                for para in p:
                    f.write(para)
            f.write(endDoc)

    paragraph_function(chapters, paragraphs, write_chapter)


def get_dtypes(dF):
    """
    This is gonna have ai to find dtype
    dF: pandas DataFrame
    :return:
    """
    m = ['Numerical' for col in dF.columns]
    return m


def render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir, heavyBreak=True):
    env = Environment(loader=FileSystemLoader(templateDir))
    template = env.get_template('jinjaTemplate.html')
    templateVars = {
        'chapterTitle':    chapterTitle, 'chapterAbstract': chapterAbstract,
        'chapterSections': chapterSections, 'heavyBreak': heavyBreak
    }
    htmlOut = template.render(templateVars)
    with open(outputPath, 'w') as f:
        f.write(htmlOut)


def randstr():
    s = ''
    for m in range(random.randint(3, 7)):
        for j in range(random.randint(6, 12)):
            if j == 0:
                s += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            k = random.randint(3, 7)
            for i in range(k):
                s += random.choice('abcdefghijklmnopqrstuvwxyz')
            s += ' '
        s += '. '
    return s


def generate_sentences(chapterSections):
    sections = chapterSections
    for section in sections:
        sec = sections[section]
        abstract = generate_textStruct(sec['abstract'])
        sec['abstract'] = abstract
        for subSec in sec['subSections']:
            sS = sec['subSections'][subSec]
            preText = generate_textStruct(sS['preText'])
            sections[section]['subSections'][subSec]['preText'] = preText
            for figure in sS['figures']:
                caption = generate_textStruct(sS['figures'][figure]['caption'])
                sections[section]['subSections'][subSec]['figures'][figure]['caption'] = caption
            postText = generate_textStruct(sS['postText'])
            sections[section]['subSections'][subSec]['postText'] = postText
    return sections


def generate_textStruct(textStruct):
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


def render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir, heavyBreak=True):
    abstract = generate_textstruct(chapterAbstract)
    sections = generate_sentences(chapterSections)
    env = Environment(loader=FileSystemLoader(templateDir))
    template = env.get_template('jinjaTemplate.html')
    templateVars = {
        'chapterTitle': chapterTitle, 'chapterAbstract': abstract, 'chapterSections': sections,
        'heavyBreak':   heavyBreak
    }

    htmlOut = template.render(templateVars)
    with open(outputPath, 'w') as f:
        f.write(htmlOut)


def generate_sentences(chapterSections):
    sections = chapterSections
    for section in sections:
        sec = sections[section]
        abstract = generate_textstruct(sec['abstract'])
        sec['abstract'] = abstract
        for subSec in sec['subSections']:
            sS = sec['subSections'][subSec]
            for figure in sS['preFigures']:
                caption = generate_textstruct(sS['preFigures'][figure]['caption'])
                sections[section]['subSections'][subSec]['preFigures'][figure]['caption'] = caption
            preText = generate_textstruct(sS['preText'])
            sections[section]['subSections'][subSec]['preText'] = preText
            for figure in sS['postFigures']:
                caption = generate_textstruct(sS['postFigures'][figure]['caption'])
                sections[section]['subSections'][subSec]['postFigures'][figure]['caption'] = caption
            postText = generate_textstruct(sS['postText'])
            sections[section]['subSections'][subSec]['postText'] = postText
    return sections


def generate_textstruct(textStruct):
    """
    textStruct: list of str
    return:
    """
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


def generate_textstruct(textStruct):
    """
    textStruct: list of str
    return:
    """
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


def generate_textstruct(textStruct):
    """
    textStruct: list of str
    return:
    """
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


def generate_textstruct(textStruct):
    """
    textStruct: list of str
    return:
    """
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


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
        "DISTRIBUTION":       distTOC, "CENTRAL TENDENCIES": ctTOC, "Mean": ctTOC + ".1", "Median": ctTOC + ".1",
        "Mode":               ctTOC + ".2", "DISPERSION": dispTOC, "Range": dispTOC + ".1",
        "Standard Deviation": dispTOC + ".1", "Variance": dispTOC + ".2", "Skew": dispTOC + ".3",
        "Kurtosis":           dispTOC + ".5", "CORRELATION": corrTOC
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
        "mean":         paragraphize(str('The mean of a distribution describes...')), "median": paragraphize(
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
        "variance":     paragraphize(str('The variance of a distribution describes...')), "skew": paragraphize(
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
        titlePage = "</br></br></br></br></br>" + headerize(title, 1) + "</br></br>" + headerize(
            "Submitted to:",
            2
        ) + headerize(
            client,
            3
        ) + "</br></br>" + headerize(
            "Prepared by:", 2
        ) + headerize(analyst, 3) + "</br></br>" + headerize("Date:", 2) + headerize(
            dateStr,
            3
        ) + "</br></br></br>"
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
        ctDesc = header + intros["ct"] + "\n" + get_mean_desc(mean) + "\n" + get_median_desc(
            median
        ) + "\n" + get_mode_desc(mode) + "\n"
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
        dispersionDesc = desc + intros["dispersion"] + "\n" + get_range_desc(dataRange) + "\n" + get_std_desc(
            std
        ) + "\n" + get_variance_desc(variance) + "\n" + get_skew_desc(skew) + "\n" + get_kurtosis_desc(
            kurtosis
        ) + "\n"
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


def generate(file, jobDir, server, **kwargs):
    data = DataReader.read_data(file)
    data.set_index(data.columns[0], inplace=True)  # temp
    data = DataOptimizer.optimize_dataFrame(data)
    dtypes = DataProcessor.get_dtypes(data)
    templateDir = os.path.join(os.path.join(server, 'LoParReport'), 'StatisticalAnalysisTemplates0')
    SATAnalysis.write_chapter(data=data, dtypes=dtypes, path=jobDir, jobName=kwargs['jobName'], templateDir=templateDir)
    SATDescriptiveStatistics.write_chapter(path=jobDir, jobName=kwargs['jobName'], templateDir=templateDir)
    # ErrorMetricsChapter.write_chapter(path=jobDir,
    #                                  templateDir=templateDir)
    ReportConverter.convert(fileName='Analysis', openDir=os.path.join(jobDir, kwargs['jobName']))


def generate(file, jobDir, server, **kwargs):
    data = DataProcessor.read_data(file)
    data.set_index(data.columns[0], inplace=True)  # temp
    data = DataOptimizer.optimize_dataFrame(data)
    dtypes = DataProcessor.get_dtypes(data)
    templateDir = os.path.join(os.path.join(server, 'generator0'), 'StatisticalAnalysisTemplates0')
    AnalysisChapter.write_chapter(
        data=data, dtypes=dtypes, path=jobDir, jobName=kwargs['jobName'],
        templateDir=templateDir
    )
    DescriptiveStatisticsChapter.write_chapter(path=jobDir, jobName=kwargs['jobName'], templateDir=templateDir)
    # ErrorMetricsChapter.write_chapter(path=jobDir,
    #                                  templateDir=templateDir)
    convert(fileName='Analysis', openDir=os.path.join(jobDir, kwargs['jobName']))


def generate(file, jobDir, server, **kwargs):
    data = DataProcessor.read_data(file)
    data.set_index(data.columns[0], inplace=True)  # temp
    data = DataOptimizer.optimize_dataFrame(data)
    dtypes = DataProcessor.get_dtypes(data)
    templateDir = os.path.join(os.path.join(server, 'generator0'), 'StatisticalAnalysisTemplates0')
    AnalysisChapter.write_chapter(data=data, dtypes=dtypes, path=jobDir, templateDir=templateDir)
    DescriptiveStatisticsChapter.write_chapter(path=jobDir, templateDir=templateDir)


def render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=True):
    templateVars = {
        'chapterTitle':    chapterTitle, 'chapterAbstract': chapterAbstract,
        'chapterSections': chapterSections, 'heavyBreak': heavyBreak
    }
    htmlOut = template.render(templateVars)
    with open(outputPath, 'w') as f:
        f.write(htmlOut)


def randstr():
    s = ''
    for m in range(random.randint(3, 7)):
        for j in range(random.randint(6, 12)):
            if j == 0:
                s += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            k = random.randint(3, 7)
            for i in range(k):
                s += random.choice('abcdefghijklmnopqrstuvwxyz')
            s += ' '
        s += '. '
    return s


def generate_textStruct(textStruct):
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


def generate_report_html(chapters=None, paragraphs=None):
    def write_chapter(chapter, paragraphs):
        with open(os.path.join('../../../app/data/report_generator/html/HTML2', chapter.title + '.html'), 'w') as f:
            f.write(beginDoc)
            for p in (paragraphs[chapter]):
                for para in p:
                    f.write(para)
            f.write(endDoc)

    paragraph_function(chapters, paragraphs, write_chapter)


def get_kwargs(**kwargs):
    fileNames = ['DescriptiveStatistics.html', 'ErrorMetrics.html', 'Analysis.html']
    kwargs['data'] = DataReader.read_data(kwargs['files'])
    kwargs['data'].set_index(kwargs['data'].columns[0], inplace=True)  # TODO temporary for testing
    kwargs['data'] = DataOptimizer.optimize_dataFrame(kwargs['data'])
    kwargs['dtypes'] = DataProcessor.get_dtypes(kwargs['data'])
    kwargs['templateDir'] = os.path.join(kwargs['serviceRoot'], 'report_generator0', 'StatisticalAnalysisTemplates0')
    kwargs['reportTemplatePath'] = os.path.join(kwargs['templateDir'], 'report/ReportTemplate2.html')
    kwargs['tocPath'] = os.path.join(kwargs['jobDir'], kwargs['jobName'], 'TableOfContents')
    kwargs['tocTemplatePath'] = os.path.join(
        kwargs['templateDir'],
        '../app/report_generator/table_of_contents_template.html'
    )
    kwargs['titlePath'] = os.path.join(kwargs['jobDir'], kwargs['jobName'], 'TitlePage')
    kwargs['titleTemplatePath'] = os.path.join(kwargs['templateDir'], 'report/TitlePageTemplate8.html')
    kwargs['resultPath'] = os.path.join(kwargs['jobDir'], kwargs['jobName'], kwargs['jobName'])
    kwargs['paths1'] = [os.path.join(kwargs['jobDir'], kwargs['jobName'], fileName) for fileName in fileNames]
    kwargs['paths2'] = ['{}.html'.format(kwargs['titlePath']), '{}.html'.format(kwargs['tocPath']),
                        '{}.html'.format(kwargs['resultPath'])]
    kwargs['openDir'] = os.path.join(kwargs['jobDir'], kwargs['jobName'])
    return kwargs


class _Section(OrderedDict):
    def __repr__(self):
        r = 'Section {}\n'.format(self['title'])
        for key in self:
            if key != 'subsections' and key != 'title':
                r += '\t  {0}: {1}\n'.format(key, self[key])
        if self['subsections'].__len__() > 0:
            for subsection in self['subsections']:
                r += '\t\t{}'.format(self['subsections'][subsection])
        return r


def generate_report(**kwargs):
    """
    kwargs: dict,
        files, jobDir, serviceRoot
    :return:
    """
    kw = get_kwargs(**kwargs)
    SATDescriptiveStatistics.write_chapter(templatePath=kw['reportTemplatePath'], **kw)
    SATErrorMetrics.write_chapter(templatePath=kw['reportTemplatePath'], **kw)
    SATAnalysis.write_chapter(templatePath=kw['reportTemplatePath'], **kw)
    ReportMerger.merge_chapters(paths=kw['paths1'], **kw)
    SATTableOfContents.write_chapter(
        paths=kw['paths1'], outputPath=kw['tocPath'], templatePath=kw['tocTemplatePath'],
        **kw
    )
    SATTitlePage.write_chapter(outputPath=kw['titlePath'], templatePath=kw['titleTemplatePath'], **kw)
    ReportMerger.merge_chapters(paths=kw['paths2'], **kw)
    ReportConverter.convert(fileName=kw['resultPath'], **kw)


def get_kwargs(**kwargs):
    fileNames = ['DescriptiveStatistics.html', 'ErrorMetrics.html', 'Analysis.html']
    kwargs['data'] = Reader.read_data(kwargs['files'])
    print(kwargs['files'], kwargs['data'])
    kwargs['data'].set_index(kwargs['data'].columns[0], inplace=True)  # TODO temporary for testing
    kwargs['data'] = Optimizer.optimize_dataFrame(kwargs['data'])
    kwargs['dtypes'] = Processor.get_dtypes(kwargs['data'])
    kwargs['templateDir'] = os.path.join('report_generator0', 'StatisticalAnalysisTemplates0')
    kwargs['reportTemplatePath'] = os.path.join(kwargs['templateDir'], 'report/ReportTemplate1.html')
    kwargs['tocPath'] = os.path.join(kwargs['jobDir'], kwargs['jobName'], 'TableOfContents')
    kwargs['tocTemplatePath'] = os.path.join(kwargs['templateDir'], 'TableOfContentsTemplate.html')
    kwargs['titlePath'] = os.path.join(kwargs['jobDir'], kwargs['jobName'], 'TitlePage')
    kwargs['titleTemplatePath'] = os.path.join(kwargs['templateDir'], 'TitlePageTemplate.html')
    kwargs['resultPath'] = os.path.join(kwargs['jobDir'], kwargs['jobName'], kwargs['jobName'])
    kwargs['paths1'] = [os.path.join(kwargs['jobDir'], kwargs['jobName'], fileName) for fileName in fileNames]
    kwargs['paths2'] = ['{}.html'.format(kwargs['titlePath']), '{}.html'.format(kwargs['tocPath']),
                        '{}.html'.format(kwargs['resultPath'])]
    kwargs['openDir'] = os.path.join(kwargs['jobDir'], kwargs['jobName'])
    return kwargs


class _Subsection(OrderedDict):
    def __repr__(self):
        r = 'Subsection {}\n'.format(self['title'])
        for key in self:
            if key != 'content' and key != 'title':
                r += '\t\t  {0}: {1}\n'.format(key, self[key])
            elif key != 'title':
                r += '\t\t  content: '
                for con in self[key]:
                    r += ' {} '.format(str(type(self[key][con]))[25:-2])
                r += '\n'
        return r


def generate_report(**kwargs):
    """
    kwargs: dict,
        files, jobDir, serviceRoot
    :return:
    """
    kw = get_kwargs(**kwargs)

    DescriptiveStatistics.write_chapter(templatePath=kw['reportTemplatePath'], **kw)
    ErrorMetrics.write_chapter(templatePath=kw['reportTemplatePath'], **kw)
    Analysis.write_chapter(templatePath=kw['reportTemplatePath'], **kw)
    Merger.merge_chapters(paths=kw['paths1'], **kw)
    TableOfContents.write_chapter(
        paths=kw['paths1'], outputPath=kw['tocPath'], templatePath=kw['tocTemplatePath'],
        **kw
    )
    TitlePage.write_chapter(outputPath=kw['titlePath'], templatePath=kw['titleTemplatePath'], **kw)
    Merger.merge_chapters(paths=kw['paths2'], **kw)
    """Converter.convert(fileName=kw['resultPath'],
                            **kw)"""


def randstr():
    s = ''
    for m in range(random.randint(3, 7)):
        for j in range(random.randint(6, 12)):
            if j == 0:
                s += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            k = random.randint(3, 7)
            for i in range(k):
                s += random.choice('abcdefghijklmnopqrstuvwxyz')
            s += ' '
        s += '. '
    return s


def generate_textStruct(textStruct):
    text = []
    for p in textStruct:
        paragraph = ''
        for i, s in enumerate(textStruct[p]):
            length = len(textStruct[p][s]) - 1
            r = random.randint(0, length)
            sentence = textStruct[p][s][r]
            paragraph += sentence
        text.append(paragraph)
    return text


def write_chapter(template, **kwargs):
    template.write_chapter(**kwargs)


class ReportBase():

    def __init__(self, title):
        self.title = title
        self.abstract = ''

    def set_abstract(self, text):
        self.abstract = text


class Chapter(ReportBase):

    def __init__(self, title):
        ReportBase.__init__(self, title)
        self.sections = {}

    def __repr__(self):
        return 'Chapter(title="' + self.title + '",' \
                                                ' sections=' + str(self.sections) + ')'

    def set_sections(self, sections):
        for section in sections:
            self.add_section(section)

    def add_section(self, section):
        if type(section) == type(''):
            self.sections[section] = Section(section)
        elif type(section) == type(Section()):
            self.sections[section.title] = section


class _Figure(OrderedDict):
    pass


class Section(ReportBase):

    def __init__(self, title=''):
        ReportBase.__init__(self, title)
        self.subSections = {}

    def __repr__(self):
        return 'Section(title="' + self.title + '",' \
                                                ' subSections=' + str(self.subSections) + ')'

    def set_subSections(self, subSections):
        for subSection in subSections:
            self.subSections[subSection] = SubSection(subSection)

    def add_subSection(self, subSection):
        self.subSections[subSection] = SubSection(subSection)


class SubSection(ReportBase):

    def __init__(self, title=''):
        ReportBase.__init__(self, title)
        self.text = {}
        self.images = {}
        self.captions = {}

    def __repr__(self):
        return 'SubSection(title="' + self.title + '",' \
                                                   ' text=' + str(self.text) + ',' \
                                                                               ' images=' + str(self.images) + ',' \
                                                                                                               ' captions=' + str(
            self.captions
        ) + ')'

    def set_text(self, module, section, subSection):
        """
        :param text: list of strings, text to display
        :return:
        """
        text = []
        _subSection = getattr(module, section)[subSection]
        for sentence in _subSection:  # browse through sentences
            text.append('')
            r = random.randint(0, len(sentence) - 1)  # select random variation
            text[-1] += sentence[r]
        self.text = text

    def set_images(self, paths, captions):
        """
        :param paths: list of strings, paths to image files
        :return:
        """
        self.images = paths

    def set_captions(self, captions):
        """
        :param captions: list of strings, captions to accompany the images
        :return:
        """
        self.captions = captions


class _Figure(OrderedDict):
    pass


class _Caption(OrderedDict):
    pass


class _Paragraph(OrderedDict):
    def __repr__(self):
        r = 'Paragraph:\n'
        for con in self:
            r += '\t{0}: {1}\n'.format(con, self[con])
        return r


def Figure(title, image=''):
    """
    title: str,
    image: str, path to image
    """
    _f = [('title', title), ('image', image)]
    f = _Figure(_f)
    return f


def Caption(title, text):
    _p = [('title', title)]
    _p.extend([('sentence {}'.format(i), '') for i in range(n)])
    p = _Paragraph(_p)
    return p


def Paragraph(title, n=1):
    _p = [('title', title)]
    _p.extend([('sentence {}'.format(i), '') for i in range(n)])
    p = _Paragraph(_p)
    return p


class _Chapter(OrderedDict):

    def __repr__(self):
        r = 'Chapter {}\n'.format(self['title'])
        for key in self:
            if key != 'sections' and key != 'title':
                r += '   {0}: {1}\n'.format(key, self[key])
        if self['sections'].__len__() > 0:
            for section in self['sections']:
                r += '\t{}'.format(self['sections'][section])
        return r

    def insert_sections(self, sections):
        newSections = []
        for section in self['sections']:
            newSections.append((section, self['sections'][section]))
        for section in sections:
            newSections.append((section['title'], section))
        self['sections'] = OrderedDict(newSections)


class _Section(OrderedDict):

    def __repr__(self):
        r = 'Section {}\n'.format(self['title'])
        for key in self:
            if key != 'subsections' and key != 'title':
                r += '\t  {0}: {1}\n'.format(key, self[key])
        if self['subsections'].__len__() > 0:
            for subsection in self['subsections']:
                r += '\t\t{}'.format(self['subsections'][subsection])
        return r

    def insert_subsections(self, subsections):
        newSubsections = []
        for subsection in self['subsections']:
            newSubsections.append((subsection, self['subsections'][subsection]))
        for subsection in subsections:
            newSubsections.append((subsection['title'], subsection))
        self['subsections'] = OrderedDict(newSubsections)


class _Subsection(OrderedDict):

    def __repr__(self):
        r = 'Subsection {}\n'.format(self['title'])
        for key in self:
            if key != 'content' and key != 'title':
                r += '\t\t  {0}: {1}\n'.format(key, self[key])
            elif key != 'title':
                r += '\t\t  content: '
                for con in self[key]:
                    r += ' {} '.format(str(type(self[key][con]))[25:-2])
                r += '\n'
        return r

    def insert_content(self, content):
        newContent = []
        for con in self['content']:
            newContent.append((con['title'], self['content'][con]))
        for con in content:
            newContent.append((con['title'], con))
        self['content'] = OrderedDict(newContent)


def Chapter(title, sections=[]):
    _c = [('title', title), ('sections', OrderedDict([(section['title'], section) for section in sections]))]
    c = _Chapter(_c)
    return c


def Section(title, breakAfter=True, displayTitle=True, subsections=[]):
    _s = [('title', title), ('breakAfter', breakAfter), ('displayTitle', displayTitle),
          ('subsections', OrderedDict([(subsection['title'], subsection) for subsection in subsections]))]
    s = _Section(_s)
    return s


def Subsection(title, breakAfter=True, column='center', displayTitle=True, hide=False):
    _ss = [('title', title), ('breakAfter', breakAfter), ('column', column), ('content', []),
           ('displayTitle', displayTitle), ('hide', hide), ]
    ss = _Subsection(_ss)
    return ss


class _Figure(OrderedDict):
    pass


class _Caption(OrderedDict):
    pass


class _Paragraph(OrderedDict):
    def __repr__(self):
        r = 'Paragraph:\n'
        for con in self:
            r += '\t{0}: {1}\n'.format(con, self[con])
        return r


class _Chapter(OrderedDict):
    def __repr__(self):
        r = 'Chapter {}\n'.format(self['title'])
        for key in self:
            if key != 'sections' and key != 'title':
                r += '   {0}: {1}\n'.format(key, self[key])
        if self['sections'].__len__() > 0:
            for section in self['sections']:
                r += '\t{}'.format(self['sections'][section])
        return r


class _Section(OrderedDict):
    def __repr__(self):
        r = 'Section {}\n'.format(self['title'])
        for key in self:
            if key != 'subsections' and key != 'title':
                r += '\t  {0}: {1}\n'.format(key, self[key])
        if self['subsections'].__len__() > 0:
            for subsection in self['subsections']:
                r += '\t\t{}'.format(self['subsections'][subsection])
        return r


def Section(title, breakAfter=True, displayTitle=True, subsections=[]):
    _s = [('title', title), ('displayTitle', displayTitle),
          ('subsections', OrderedDict([(subsection['title'], subsection) for subsection in subsections]))]
    s = _Section(_s)
    return s


def generate_report_html(chapters=None, paragraphs=None):
    def write_chapter(chapter, paragraphs):
        with open(os.path.join('HTML', chapter.title + '.html'), 'w') as f:
            f.write(beginDoc)
            for p in (paragraphs[chapter]):
                for para in p:
                    f.write(para)
            f.write(endDoc)

    paragraph_function(chapters, paragraphs, write_chapter)


def analysis(data=None, **kwargs):
    """
    vars: list of strings
    dtypes: list of strings
    return:
    """
    imagesPath = os.path.join(kwargs['jobDir'], gc.jobImgPath)
    vars, numStats, catStats, distFit, regression = plot_api.generate_all_plots(data, kwargs['dtypes'], path=imagesPath)
    plots = OrderedDict(
        [(var, [('Scatter Plots', 'Regression Error Tables'),
                ('Distribution Plots', 'Central Tendencies Tables'),
                ('Distribution Fit Plots', 'Distribution Error Tables'), ('Box Plots', 'IQR Tables'),
                ('Violin Plots', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar Plots', '')])
         for var, dtype in zip(vars, kwargs['dtypes'])]
    )
    captions = {}
    text = {}

    captions['Bar Plot'] = {}
    text['Bar Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Categorical':
            captions['Bar Plot'][var] = 'Bar Plot of <var>{}</var>.'.format(var)
            text['Bar Plot'][var] = ''

    captions['Box Plot'] = {}
    text['Box Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            captions['Box Plot'][var] = 'Box plot of <var>{}</var>.'.format(var)
            lowerIQR = numStats[var]['Lower IQR']
            upperIQR = numStats[var]['Upper IQR']
            lowerOutliers = numStats[var]['Lower Outliers']
            upperOutliers = numStats[var]['Upper Outliers']
            maximum = numStats[var]['Max']
            minimum = numStats[var]['Min']
            quartile1 = numStats[var]['First Quartile']
            quartile3 = numStats[var]['Third Quartile']
            iqr = numStats[var]['IQR']
            boxText = 'The minimum value found in <var>{}</var> is <em>{}</em>. '.format(var, str(minimum))
            boxText += 'The maximum is {}. '.format(str(maximum))
            boxText += 'The first quartile is {} and the third quartile is {}. '.format(quartile1, quartile3)
            boxText += 'The interquartile range is therefore {}. '.format(iqr)
            boxText += 'The lower limit of the interquartile range is {}. '.format(lowerIQR)
            boxText += 'and the upper limit is {}. '.format(upperIQR)
            if upperIQR > maximum:
                boxText += 'This is above the maximum value in the data. '
            else:
                boxText += 'This is below the maximum value in the data. '
            if len(lowerOutliers) > 0:
                boxText += 'There are {} outliers below the lower bound. '.format(len(lowerOutliers))
            else:
                boxText += 'There are no outliers below the lower bound. '
            if len(upperOutliers) > 0:
                boxText += 'There are {} outliers above the upper bound. '.format(len(upperOutliers))
            else:
                boxText += 'There are no outliers above the upper bound. '
            text['Box Plot'][var] = boxText

    captions['Distribution Plot'] = {}
    text['Distribution Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            distributionText = ''
            captions['Distribution Plot'][var] = 'Histogram of <var>{}</var>.'.format(var)
            text['Distribution Plot'][var] = distributionText

    captions['Distribution Fit Plot'] = {}
    text['Distribution Fit Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            keys = list(distFit[var].keys())
            func1 = distFit[var][keys[0]][0]
            # func2 = distFit[var][keys[1]][0]
            error1 = util.round_array(distFit[var][keys[0]][1])
            # error2 = s_round(distFit[var][keys[1]][1])
            captions['Distribution Fit Plot'][var] = 'Distribution fit of <var>{0}</var>.'.format(var)
            # Distribution functions
            distributionFitText = r'$$\textbf{' + keys[0] + ':}$$' + '$${}$$'.format(
                func1
            )  # + r'$$\textbf{' + keys[1] + ':}$$' + '$${}$$'.format(func2) ])
            text['Distribution Fit Plot'][var] = distributionFitText

    captions['Scatter Plot'] = {}
    text['Scatter Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            fit = util.round_array(regression[var][0])
            errors = regression[var][1]
            captions['Scatter Plot'][var] = 'Scatter plot and regression fit of <var>{}</var>.'.format(var)
            scatterText = '<var>{}</var> is best approximated by the following function: $${}$$ '.format(var, fit)
            text['Scatter Plot'][var] = scatterText

    captions['Violin Plot'] = {}
    text['Violin Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            captions['Violin Plot'][var] = 'Violin plot of <var>{0}</var>.'.format(var)
            mean = util.round_array(numStats[var]['Mean'])
            median = util.round_array(numStats[var]['Median'])
            std = util.round_array(numStats[var]['Standard Deviation'])
            variance = util.round_array(numStats[var]['Variance'])
            skew = util.round_array(numStats[var]['Skew'])
            kurtosis = util.round_array(numStats[var]['Kurtosis'])
            x = util.round_array(mean - median)
            c0 = util.round_array(mean / median)
            c1 = util.round_array((abs(x) / mean) * 100)
            if x > 0:  # mean > median
                meanMedianCompare = 'The <em>mean</em> is {} times, or {}% greater than the <em>median</em>. '.format(
                    str(c0), str(c1)
                )
            else:  # mean < median
                meanMedianCompare = 'The <em>mean</em> is {} times, or {}% smaller than the <em>median</em>. '.format(
                    str(util.round_array(1 / c0)), str(c1)
                )
            # STANDARD DEVIATION --------------------------------
            stdText = 'This distribution has a <em>standard deviation</em> of {}. '.format(str(std))
            # VARIANCE ------------------------------------------
            varianceText = 'Therefore, the <em>variance</em> is {}. '.format(str(variance))
            # SKEW ----------------------------------------------
            skewText = 'The distribution <em>skew</em>  is {}, which indicates that the data is concentrated '.format(
                skew
            )
            if abs(skew) <= 0.5:
                skewText += 'slightly'
            elif abs(skew) <= 1:
                skewText += 'moderately'
            else:
                skewText += 'highly'
            if skew > 0:
                skewText += ' below the mean with a longer tail above the mean. '
            elif skew < 0:
                skewText += ' above the mean with a longer tail below the mean. '
            else:
                skewText += ' centered with symmetrical tails.'
            # KURTOSIS ------------------------------------------
            kurtosisText = 'The <em>kurtosis</em> of {} means that the data is '.format(kurtosis)
            if kurtosis < 0:
                kurtosisText += ' a very spread out<em>platykurtic</em> distribution. The tails of the distribution' \
                                ' are relatively long, slender, and more prone to contain outliers. '
            elif kurtosis > 0:
                kurtosisText += ' a tightly concentrated<em>leptokurtic</em> distribution. The tails of the' \
                                '  distribution are relatively short, broad, and less prone to contain outliers. '
            else:
                kurtosisText += ' a moderately spread out <em>mesokurtic</em> distribution. '
            violinText = '{} {} {} {} {}'.format(meanMedianCompare, stdText, varianceText, skewText, kurtosisText)
            text['Violin Plot'][var] = violinText

    captions['Correlation Plot'] = {}
    if 'Numerical' in kwargs['dtypes']:
        captions['Correlation Plot'] = 'Correlation heatmap. '
        correlationText = 'This is the correlation matrix of the data plotted as a heatmap. '
        text['Correlation Plot'] = correlationText

    chapterSections = OrderedDict(
        [  # SECTION ----------------------
            (var, {  # SUBSECTIONS --------------
                'subSections': OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'figures': {
                                'Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ): {
                                    'image':   os.path.join(jobImgPath, plot[0].replace(' ', '')[:-5], var + '.png'),
                                    'caption': '<var>Figure {}:</var> {}'.format(
                                        (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]),
                                        captions[plot[0][:-1]][var]
                                    )
                                },
                                'Figure {}'.format(
                                    (i + 2) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ): {
                                    'image':   os.path.join(jobImgPath, plot[1].replace(' ', '')[:-6], var + '.png'),
                                    'caption': '<var>Table {0}:</var> {1}'.format(
                                        (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]),
                                        captions[plot[0][:-1]][var]
                                    )
                                },
                            }, 'text': text[plot[0][:-1]][var],
                        }) for i, plot in
                        enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )
    if 'Numerical' in kwargs['dtypes']:
        chapterSections.update(
            {
                'Correlation': {  # SUBSECTIONS --------------
                    'subSections': OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'figures': {
                                    'Figure -1': {
                                        'image':   os.path.join(jobImgPath, 'Heatmap', 'Correlation.png'),
                                        'caption': '<var>Figure {}:</var> {}'.format(
                                            1 + sum([len(plots[v]) for v in vars]),
                                            captions['Correlation Plot']
                                        )
                                    },
                                },
                                'text':    text['Correlation Plot'],
                            })]
                    )
                }
            }
        )
    chapterSections.move_to_end('Correlation', last=True)
    templateVars = {'chapterTitle': 'Analysis', 'chapterSections': chapterSections}
    render_HTML(templatePath=gc.analysisTemplatePath, templateVars=templateVars, outputPath=kwargs['analysisPath'])


def merge(soups):
    def _merge(soup1, soup2):
        for element in soup2:
            soup1.body.append(element)
        return soup1

    for i, soup in enumerate(soups):
        if i != 0:
            soups[0] = _merge(soups[0], soup)
    return str(soups[0].body)


def get_kwargs():
    jobID = util.generate_job_id()
    kwargs = {'jobDate': '{0} {1}, {2}'.format(calendar.month_name[gc.now.month], gc.now.day, gc.now.year)}
    kwargs['jobDir'] = os.path.join(gc.jobRoot, jobID)
    kwargs['navPath'] = os.path.join(kwargs['jobDir'], 'navigator.html')
    kwargs['resultPath'] = os.path.join(kwargs['jobDir'], 'result.html')
    kwargs['analysisPath'] = os.path.join(kwargs['jobDir'], 'analysis.html')
    kwargs['paths1'] = [kwargs['navPath'], kwargs['resultPath']]
    kwargs['plotsDir'] = os.path.join(kwargs['jobDir'], 'plots')
    return kwargs


def analysis(data=None, **kwargs):
    """
    vars: list of strings
    dtypes: list of strings
    return:
    """
    imagesPath = os.path.join(kwargs['jobDir'], gc.jobImgPath)
    vars, numStats, catStats, distFit, regression = plot_api.generate_all_plots(data, kwargs['dtypes'], path=imagesPath)
    plots = OrderedDict(
        [(var, [('Scatter Plots', 'Regression Error Tables'),
                ('Distribution Plots', 'Central Tendencies Tables'),
                ('Distribution Fit Plots', 'Distribution Error Tables'), ('Box Plots', 'IQR Tables'),
                ('Violin Plots', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar Plots', '')])
         for var, dtype in zip(vars, kwargs['dtypes'])]
    )
    captions = {}
    text = {}

    captions['Bar Plot'] = {}
    text['Bar Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Categorical':
            captions['Bar Plot'][var] = 'Bar Plot of <var>{}</var>.'.format(var)
            text['Bar Plot'][var] = ''

    captions['Box Plot'] = {}
    text['Box Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            captions['Box Plot'][var] = 'Box plot of <var>{}</var>.'.format(var)
            lowerIQR = numStats[var]['Lower IQR']
            upperIQR = numStats[var]['Upper IQR']
            lowerOutliers = numStats[var]['Lower Outliers']
            upperOutliers = numStats[var]['Upper Outliers']
            maximum = numStats[var]['Max']
            minimum = numStats[var]['Min']
            quartile1 = numStats[var]['First Quartile']
            quartile3 = numStats[var]['Third Quartile']
            iqr = numStats[var]['IQR']
            boxText = 'The minimum value found in <var>{}</var> is <em>{}</em>. '.format(var, str(minimum))
            boxText += 'The maximum is {}. '.format(str(maximum))
            boxText += 'The first quartile is {} and the third quartile is {}. '.format(quartile1, quartile3)
            boxText += 'The interquartile range is therefore {}. '.format(iqr)
            boxText += 'The lower limit of the interquartile range is {}. '.format(lowerIQR)
            boxText += 'and the upper limit is {}. '.format(upperIQR)
            if upperIQR > maximum:
                boxText += 'This is above the maximum value in the data. '
            else:
                boxText += 'This is below the maximum value in the data. '
            if len(lowerOutliers) > 0:
                boxText += 'There are {} outliers below the lower bound. '.format(len(lowerOutliers))
            else:
                boxText += 'There are no outliers below the lower bound. '
            if len(upperOutliers) > 0:
                boxText += 'There are {} outliers above the upper bound. '.format(len(upperOutliers))
            else:
                boxText += 'There are no outliers above the upper bound. '
            text['Box Plot'][var] = boxText

    captions['Distribution Plot'] = {}
    text['Distribution Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            distributionText = ''
            captions['Distribution Plot'][var] = 'Histogram of <var>{}</var>.'.format(var)
            text['Distribution Plot'][var] = distributionText

    captions['Distribution Fit Plot'] = {}
    text['Distribution Fit Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            keys = list(distFit[var].keys())
            func1 = distFit[var][keys[0]][0]
            # func2 = distFit[var][keys[1]][0]
            error1 = util.round_array(distFit[var][keys[0]][1])
            # error2 = s_round(distFit[var][keys[1]][1])
            captions['Distribution Fit Plot'][var] = 'Distribution fit of <var>{0}</var>.'.format(var)
            # Distribution functions
            distributionFitText = r'$$\textbf{' + keys[0] + ':}$$' + '$${}$$'.format(
                func1
            )  # + r'$$\textbf{' + keys[1] + ':}$$' + '$${}$$'.format(func2) ])
            text['Distribution Fit Plot'][var] = distributionFitText

    captions['Scatter Plot'] = {}
    text['Scatter Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            fit = util.round_array(regression[var][0])
            errors = regression[var][1]
            captions['Scatter Plot'][var] = 'Scatter plot and regression fit of <var>{}</var>.'.format(var)
            scatterText = '<var>{}</var> is best approximated by the following function: $${}$$ '.format(var, fit)
            text['Scatter Plot'][var] = scatterText

    captions['Violin Plot'] = {}
    text['Violin Plot'] = {}
    for var, dtype in zip(vars, kwargs['dtypes']):
        if dtype == 'Numerical':
            captions['Violin Plot'][var] = 'Violin plot of <var>{0}</var>.'.format(var)
            mean = util.round_array(numStats[var]['Mean'])
            median = util.round_array(numStats[var]['Median'])
            std = util.round_array(numStats[var]['Standard Deviation'])
            variance = util.round_array(numStats[var]['Variance'])
            skew = util.round_array(numStats[var]['Skew'])
            kurtosis = util.round_array(numStats[var]['Kurtosis'])
            x = util.round_array(mean - median)
            c0 = util.round_array(mean / median)
            c1 = util.round_array((abs(x) / mean) * 100)
            if x > 0:  # mean > median
                meanMedianCompare = 'The <em>mean</em> is {} times, or {}% greater than the <em>median</em>. '.format(
                    str(c0), str(c1)
                )
            else:  # mean < median
                meanMedianCompare = 'The <em>mean</em> is {} times, or {}% smaller than the <em>median</em>. '.format(
                    str(util.round_array(1 / c0)), str(c1)
                )
            # STANDARD DEVIATION --------------------------------
            stdText = 'This distribution has a <em>standard deviation</em> of {}. '.format(str(std))
            # VARIANCE ------------------------------------------
            varianceText = 'Therefore, the <em>variance</em> is {}. '.format(str(variance))
            # SKEW ----------------------------------------------
            skewText = 'The distribution <em>skew</em>  is {}, which indicates that the data is concentrated '.format(
                skew
            )
            if abs(skew) <= 0.5:
                skewText += 'slightly'
            elif abs(skew) <= 1:
                skewText += 'moderately'
            else:
                skewText += 'highly'
            if skew > 0:
                skewText += ' below the mean with a longer tail above the mean. '
            elif skew < 0:
                skewText += ' above the mean with a longer tail below the mean. '
            else:
                skewText += ' centered with symmetrical tails.'
            # KURTOSIS ------------------------------------------
            kurtosisText = 'The <em>kurtosis</em> of {} means that the data is '.format(kurtosis)
            if kurtosis < 0:
                kurtosisText += ' a very spread out<em>platykurtic</em> distribution. The tails of the distribution' \
                                ' are relatively long, slender, and more prone to contain outliers. '
            elif kurtosis > 0:
                kurtosisText += ' a tightly concentrated<em>leptokurtic</em> distribution. The tails of the' \
                                '  distribution are relatively short, broad, and less prone to contain outliers. '
            else:
                kurtosisText += ' a moderately spread out <em>mesokurtic</em> distribution. '
            violinText = '{} {} {} {} {}'.format(meanMedianCompare, stdText, varianceText, skewText, kurtosisText)
            text['Violin Plot'][var] = violinText

    captions['Correlation Plot'] = {}
    if 'Numerical' in kwargs['dtypes']:
        captions['Correlation Plot'] = 'Correlation heatmap. '
        correlationText = 'This is the correlation matrix of the data plotted as a heatmap. '
        text['Correlation Plot'] = correlationText

    chapterSections = OrderedDict(
        [  # SECTION ----------------------
            (var, {  # SUBSECTIONS --------------
                'subSections': OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'figures': {
                                'Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ): {
                                    'image':   os.path.join(jobImgPath, plot[0].replace(' ', '')[:-5], var + '.png'),
                                    'caption': '<var>Figure {}:</var> {}'.format(
                                        (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]),
                                        captions[plot[0][:-1]][var]
                                    )
                                },
                                'Figure {}'.format(
                                    (i + 2) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ): {
                                    'image':   os.path.join(jobImgPath, plot[1].replace(' ', '')[:-6], var + '.png'),
                                    'caption': '<var>Table {0}:</var> {1}'.format(
                                        (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]),
                                        captions[plot[0][:-1]][var]
                                    )
                                },
                            }, 'text': text[plot[0][:-1]][var],
                        }) for i, plot in
                        enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )
    if 'Numerical' in kwargs['dtypes']:
        chapterSections.update(
            {
                'Correlation': {  # SUBSECTIONS --------------
                    'subSections': OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'figures': {
                                    'Figure -1': {
                                        'image':   os.path.join(jobImgPath, 'Heatmap', 'Correlation.png'),
                                        'caption': '<var>Figure {}:</var> {}'.format(
                                            1 + sum([len(plots[v]) for v in vars]),
                                            captions['Correlation Plot']
                                        )
                                    },
                                },
                                'text':    text['Correlation Plot'],
                            })]
                    )
                }
            }
        )
    chapterSections.move_to_end('Correlation', last=True)
    templateVars = {'chapterTitle': 'Analysis', 'chapterSections': chapterSections}
    render_HTML(templatePath=gc.analysisTemplatePath, templateVars=templateVars, outputPath=kwargs['analysisPath'])


def generate_report(directory, source='../ts_django/tensorstone_statistics/job_template'):
    kwargs = get_kwargs()
    for k in kwargs:
        print(k, kwargs[k])
    shutil.copytree(src=source, dst=kwargs['jobDir'])
    make_dirs(kwargs)
    f = util.get_files(directory)[0]
    kwargs['data'], kwargs['dtypes'] = data_api.read_data(f)
    print(kwargs['dtypes'])
    kwargs['data'].set_index(kwargs['data'].columns[0], inplace=True)  # TODO temporary for testing
    analysis(templatePath=gc.analysisTemplatePath, **kwargs)
    merge_chapters(paths=[kwargs['analysisPath']], **kwargs)
    navigator(**kwargs)
    merge_chapters(paths=kwargs['paths1'], **kwargs)


def write_chapter(templatePath='', outputPath='', jobName='', jobDate='', **kwargs):
    _outputPath = GeneralUtil.add_extension(outputPath, 'html')
    templateVars = {'jobName': jobName, 'jobDate': jobDate}
    ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=_outputPath)


def write_chapter(paths='', templatePath='', outputPath='', **kwargs):
    _paths = [GeneralUtil.add_extension(path, 'html') for path in paths]
    _outputPath = GeneralUtil.add_extension(outputPath, 'html')
    headers = collect_headers(_paths)
    headersDict = parse_headers(headers)
    templateVars = {'headersDict': headersDict}
    ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=_outputPath)


def openFile():
    oFile = open('mtguide.pdf', 'rb')
    rFile = pdf.PdfFileReader(oFile)
    # print (rFile.numPages)
    readFile(rFile)


def openFile():
    oFile = open('mtguide.pdf', 'rb')
    rFile = pdf.PdfFileReader(oFile)
    # print (rFile.numPages)
    readFile(rFile)


def readFile(rFile):
    i, j = 0, 0
    fullData, reducedData = [], []
    for line in rFile:
        fullData.append(line.rstrip())
    for i in range(fullData.__len__()):
        if fullData[i] == "DEFINITION OF TERMS":
            while fullData[i] != "DATA ITEMS":
                reducedData.append(fullData[i].rstrip())
                i += 1
        elif fullData[i] == "DATA ITEMS":
            createFile()
            return
    for i in range(fullData.__len__()):
        print(fullData[i])
    print(reducedData)


def readFile(rFile):
    i, j = 0, 0
    fullData, reducedData = [], []
    for line in rFile:
        fullData.append(line.rstrip())

    for i in range(fullData.__len__()):
        if fullData[i] == "DEFINITION OF TERMS":
            while fullData[i] != "DATA ITEMS":
                reducedData.append(fullData[i].rstrip())
                i += 1

        elif fullData[i] == "DATA ITEMS":
            createFile()
            return

    for i in range(fullData.__len__()):
        print(fullData[i])
    print(reducedData)


def createFile():
    pass  # global wFile = open("testOut.txt", "w")


def createFile():
    pass  # global wFile = open("testOut.txt", "w")


def writeFile():
    pass  # for line in data:
    # wFile.write(data[i])


def writeFile():
    pass  # for line in data:
    # wFile.write(data[i])


class BaseSection():
    def __init__(self, name, abstract='', breakAfter=True, displayTitle=True):
        self.name = name
        self.abstract = ''
        self.breakAfter = ''
        self.displayTitle = displayTitle


class _Caption(OrderedDict):
    pass


class _Chapter(OrderedDict):
    def __repr__(self):
        r = 'Chapter {}\n'.format(self['title'])
        for key in self:
            if key != 'sections' and key != 'title':
                r += '   {0}: {1}\n'.format(key, self[key])
        if self['sections'].__len__() > 0:
            for section in self['sections']:
                r += '\t{}'.format(self['sections'][section])
        return r


class Chapter():
    def __init__(self, name, abstract, sections=None):
        self.name = name
        self.abstract = Paragraph(abstract)
        self.sections = [Section(section) for section in sections]

    def __repr__(self):
        repr = 'Chapter: {}\n'.format(self.name)
        repr += 'Abstract:\n'
        repr += '\t{}\n'.format(self.abstract.text)
        repr += 'Sections:\n'
        for section in self.sections:
            repr += '\t{}\n'.format(section.name)
        repr += '\n'
        return repr

    def insert_section(self, name):
        self.sections.append(Section(name))


class Section(BaseSection):
    def __init__(self, name, abstract, breakAfter, displayTitle):
        BaseSection.__init__(self, name, abstract, breakAfter, displayTitle)


class SubSection(BaseSection):
    def __init__(self, name, abstract, breakAfter, displayTitle):
        BaseSection.__init__(self, name, abstract, breakAfter, displayTitle)


class _Paragraph(OrderedDict):
    def __repr__(self):
        r = 'Paragraph:\n'
        for con in self:
            r += '\t{0}: {1}\n'.format(con, self[con])
        return r


class Paragraph():
    def __init__(self, text=''):
        self.text = text


class _Subsection(OrderedDict):
    def __repr__(self):
        r = 'Subsection {}\n'.format(self['title'])
        for key in self:
            if key != 'content' and key != 'title':
                r += '\t\t  {0}: {1}\n'.format(key, self[key])
            elif key != 'title':
                r += '\t\t  content: '
                for con in self[key]:
                    r += ' {} '.format(str(type(self[key][con]))[25:-2])
                r += '\n'
        return r


class WriteReport:
    # WRITE REPORT
    def __init__(self, Search, year, first):
        self.Search = Search
        self.year, self.first = year, first
        if first:
            mode = "w"
        else:
            mode = "a"
        csv_export = self.Search.settings[0]
        machine_export = self.Search.settings[1]
        single_file = self.Search.settings[2]
        multiple_files = self.Search.settings[3]
        # CSV EXPORT
        if csv_export:
            csv_version_bridge = []
            with open(self.Search.GUI.filepath + "-csv_version.txt", mode) as report:
                for bridge in self.Search.final_results:  # Iterating through the bridge dictionaries in the results list
                    if '#########' not in bridge:
                        csv_version_bridge = [bridge[value] for value in bridge]
                        for item in range(csv_version_bridge.__len__()):
                            report.write((str(csv_version_bridge[item])) + ",")
                        report.write("\n")
                        csv_version_bridge[:] = []
                    else:  # Adding the indicator of a new state/year
                        bridge = re.findall(r'\w+', str(bridge))  # Converting the indicator into a list of components
                        csv_version_bridge.append(
                            str(bridge[0]).strip("#")
                        )  # Adding the result to the the list of csv results
        # MACHINE LEARNING INPUT FILE EXPORT
        if machine_export:
            if multiple_files:
                with open(self.Search.GUI.filepath + "-" + str(self.year)[2:] + ".txt", mode) as report:
                    for bridge in self.Search.final_results:
                        report.write(str(bridge) + "\n")
            if single_file:
                with open(self.Search.GUI.filepath + ".txt", mode) as report:
                    for bridge in self.Search.final_results:
                        report.write(str(bridge) + "\n")


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('/'), 'HTML2'), 'ErrorMetrics.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean Absolute Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                MAE of a data set is the average distance between
                                each data value and the mean of a data set.
                                It describes the variation and dispersion in a data set.
                                """,

                                                     """
                                                                                                       MAE of a data set measures the average distance between
                                                                                                       each data value and the mean of a data set.
                                                                                                       It describes the variation and dispersion in a data set.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    It is the average magnitude
                                    of the errors in a set of predictions. Therefore, direction is not considered.
                                    """,

                                                     """
                                                                                                       Without their direction being considered, MAE averages the magnitude
                                                                                                       of the errors in a set of predictions.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    ####
                                    It’s the average over the test sample of the absolute
                                    differences between prediction and actual observation where
                                    all individual differences have equal weight.
                                    ####
                                    """,

                                                     """
                                                                                                       ####
                                                                                                       It’s the average over the test sample of the absolute
                                                                                                       differences between prediction and actual observation where
                                                                                                       all individual differences have equal weight.
                                                                                                       """])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                            ####
                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                            """,

                                                     """
                                                                                                       ###
                                                                                                       MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                                                                                                       """]),
                                     ('sentence 1', ["""
                            MAE is calculated as follows:
                            """,

                                                     """
                                                                                                       MAE is calculated using the following formula:
                                                                                                       """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Absolute Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Bias Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             If the absolute value is not taken (the signs of the errors are not removed),
                             the average error becomes the Mean Bias Error (MBE) and is
                             usually intended to measure average model bias.
                             MBE can convey useful information, but should be interpreted cautiously
                             because positive and negative errors will cancel out.
                             ####
                             """,

                                                     """
                                                                                                                      ####
                                                                                                                      If the absolute value is not taken (the signs of the errors are not removed),
                                                                                                                      the average error becomes the Mean Bias Error (MBE) and is
                                                                                                                      usually intended to measure average model bias.
                                                                                                                      MBE can convey useful information, but should be interpreted cautiously
                                                                                                                      because positive and negative errors will cancel out.
                                                                                                                      ####
                                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Squared Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The mean squared error is calculated as follows:
                             """,

                                                     """
                                                                                                      The mean squared error is calculated using the following formula:
                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Root Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                             It’s the square root of the average of squared differences between prediction and actual observation.
                             ###
                             """,

                                                     """
                                                                                                           ####
                                                                                                           RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                                                                                                           It’s the square root of the average of squared differences between prediction and actual observation.
                                                                                                           ###
                                                                                                           """]),
                                     ('sentence 1', ["""
                             RMSE does not necessarily increase with the variance of the errors.
                             RMSE is proportional to the variance of the error magnitude frequency distribution.
                             However, it is not necessarily proporional to the error variance.
                             """,

                                                     """
                                                                                                           RMSE does not necessarily increase with the variance of the errors.
                                                                                                           RMSE is proportional to the variance of the error magnitude frequency distribution.
                                                                                                           However, it is not necessarily proporional to the error variance.
                                                                                                           """]), ]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 1', ["""
                             RMSE can be appropriate when large errors should be penalized more than smaller errors.
                             The penalty of the error changes in a non-linear way.
                             """,

                                                     """
                                                                                                           ###
                                                                                                           If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate
                                                                                                           in some cases penalizing large errors more.
                                                                                                           If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate.
                                                                                                           ####
                                                                                                           """]),
                                     ('sentence 2', ["""
                             The root mean square error is calculated as follows:
                             """,

                                                     """
                                                                                                           The root mean square error is calculated using the following formula:
                                                                                                           """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Root Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [''])]
                                ))]
                            ),
                        })]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('/'), 'HTML2'), 'Quartiles.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../../Report2/'), 'HTML2'), 'Quartiles.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(ReportPaths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(ReportPaths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', ["""
                                     <em>IQR</em> is calculated as:
                                     """])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The minimum value in the data is defined as:</br>
                             """]), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = ReportGenerator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = ReportGenerator.generate_sentences(chapterSections)
    htmlOut = ReportGenerator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../LoParReportGenerator1/'), 'HTML2'), 'Quartiles.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../report_generator0/'), 'HTML2'), 'Quartiles.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(ReportPaths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(ReportPaths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', ["""
                                     <em>IQR</em> is calculated as:
                                     """])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The minimum value in the data is defined as:</br>
                             """]), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = ReportGenerator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = ReportGenerator.generate_sentences(chapterSections)
    htmlOut = ReportGenerator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(
        os.path.join(os.path.abspath('../../generator/ReportGenerator0/'), 'HTML2'), 'Quartiles.html'
    )

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(ReportPaths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(ReportPaths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', ["""
                                     <em>IQR</em> is calculated as:
                                     """])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The minimum value in the data is defined as:</br>
                             """]), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = ReportGenerator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = ReportGenerator.generate_sentences(chapterSections)
    htmlOut = ReportGenerator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../../generator/generator0/'), 'HTML2'), 'Quartiles.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(
        os.path.join(os.path.abspath('../../generator/LoParReportGenerator3/'), 'HTML2'), 'Quartiles.html'
    )

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../../LoParReportGenerator/'), 'HTML2'), 'Quartiles.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', ["""
                                     <em>IQR</em> is calculated as:
                                     """])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The minimum value in the data is defined as:</br>
                             """]), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../../generator/LoParReport/'), 'HTML2'), 'Quartiles.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Quartiles', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The <var>1st Quartile</var> is the <em>25th Percentile</em>.
                                This is the value below which 25% of the data exists,
                                and above which 75% of the data exists.
                                """, ]), ('sentence 1', ["""
                                The <var>2nd Quartile</var> is the <em>50th Percentile</em>.
                                This is the value below which
                                50% of the data is above and 50% is below.
                                This value is more commonly refered to as the <var>Median</var>.
                                """, ]), ('sentence 1', ["""
                                    ####
                                    The <var>3rd Quartile</var> is the <em>75th Percentile</em>.
                                    This is the value below which 75% of the data exists,
                                    and above which 25% of the data exists.
                                    ####
                                    """, ])]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Absolute Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }), ]
                )
            }),

            ('Interquartile Range', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The <em>IQR</em> is used to calculate <var>outliers</var>.
                            Any value that is a distance of <var>1.5×IQR</var> <em>below the 1st Quartile or
                            above the 3rd Quartile</em> is considered an outlier.
                            This changes what how the minimum and maximum values are defined.
                             """, ]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', ["""
                                     <em>IQR</em> is calculated as:
                                     """])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The minimum value in the data is defined as:</br>
                             """]), ]
                                )), ]
                            ),
                        }), ]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    chapter = ReportGenerator.Chapter('Descriptive Statistics')
    mean = mean_section()
    median = median_section()
    mode = mode_section()
    skew = skew_section()
    kurtosis = kurtosis_section()
    chapter.insert_sections([mean])
    print(chapter)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(
        os.path.join(os.path.abspath('../../generator/generator0/'), 'HTML2'), 'ErrorMetrics.html'
    )

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean Absolute Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                MAE of a data set is the average distance between
                                each data value and the mean of a data set.
                                It describes the variation and dispersion in a data set.
                                """,

                                                     """
                                                                                                       MAE of a data set measures the average distance between
                                                                                                       each data value and the mean of a data set.
                                                                                                       It describes the variation and dispersion in a data set.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    It is the average magnitude
                                    of the errors in a set of predictions. Therefore, direction is not considered.
                                    """,

                                                     """
                                                                                                       Without their direction being considered, MAE averages the magnitude
                                                                                                       of the errors in a set of predictions.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    ####
                                    It’s the average over the test sample of the absolute
                                    differences between prediction and actual observation where
                                    all individual differences have equal weight.
                                    ####
                                    """,

                                                     """
                                                                                                       ####
                                                                                                       It’s the average over the test sample of the absolute
                                                                                                       differences between prediction and actual observation where
                                                                                                       all individual differences have equal weight.
                                                                                                       """])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                            ####
                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                            """,

                                                     """
                                                                                                       ###
                                                                                                       MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                                                                                                       """]),
                                     ('sentence 1', ["""
                            MAE is calculated as follows:
                            """,

                                                     """
                                                                                                       MAE is calculated using the following formula:
                                                                                                       """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Absolute Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Bias Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             If the absolute value is not taken (the signs of the errors are not removed),
                             the average error becomes the Mean Bias Error (MBE) and is
                             usually intended to measure average model bias.
                             MBE can convey useful information, but should be interpreted cautiously
                             because positive and negative errors will cancel out.
                             ####
                             """,

                                                     """
                                                                                                                      ####
                                                                                                                      If the absolute value is not taken (the signs of the errors are not removed),
                                                                                                                      the average error becomes the Mean Bias Error (MBE) and is
                                                                                                                      usually intended to measure average model bias.
                                                                                                                      MBE can convey useful information, but should be interpreted cautiously
                                                                                                                      because positive and negative errors will cancel out.
                                                                                                                      ####
                                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Squared Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The mean squared error is calculated as follows:
                             """,

                                                     """
                                                                                                      The mean squared error is calculated using the following formula:
                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Root Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                             It’s the square root of the average of squared differences between prediction and actual observation.
                             ###
                             """,

                                                     """
                                                                                                           ####
                                                                                                           RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                                                                                                           It’s the square root of the average of squared differences between prediction and actual observation.
                                                                                                           ###
                                                                                                           """]),
                                     ('sentence 1', ["""
                             RMSE does not necessarily increase with the variance of the errors.
                             RMSE is proportional to the variance of the error magnitude frequency distribution.
                             However, it is not necessarily proporional to the error variance.
                             """,

                                                     """
                                                                                                           RMSE does not necessarily increase with the variance of the errors.
                                                                                                           RMSE is proportional to the variance of the error magnitude frequency distribution.
                                                                                                           However, it is not necessarily proporional to the error variance.
                                                                                                           """]), ]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 1', ["""
                             RMSE can be appropriate when large errors should be penalized more than smaller errors.
                             The penalty of the error changes in a non-linear way.
                             """,

                                                     """
                                                                                                           ###
                                                                                                           If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate
                                                                                                           in some cases penalizing large errors more.
                                                                                                           If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate.
                                                                                                           ####
                                                                                                           """]),
                                     ('sentence 2', ["""
                             The root mean square error is calculated as follows:
                             """,

                                                     """
                                                                                                           The root mean square error is calculated using the following formula:
                                                                                                           """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Root Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [''])]
                                ))]
                            ),
                        })]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(
        os.path.join(os.path.abspath('../../generator/LoParReportGenerator3/'), 'HTML2'), 'ErrorMetrics.html'
    )

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean Absolute Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                MAE of a data set is the average distance between
                                each data value and the mean of a data set.
                                It describes the variation and dispersion in a data set.
                                """,

                                                     """
                                                                                                       MAE of a data set measures the average distance between
                                                                                                       each data value and the mean of a data set.
                                                                                                       It describes the variation and dispersion in a data set.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    It is the average magnitude
                                    of the errors in a set of predictions. Therefore, direction is not considered.
                                    """,

                                                     """
                                                                                                       Without their direction being considered, MAE averages the magnitude
                                                                                                       of the errors in a set of predictions.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    ####
                                    It’s the average over the test sample of the absolute
                                    differences between prediction and actual observation where
                                    all individual differences have equal weight.
                                    ####
                                    """,

                                                     """
                                                                                                       ####
                                                                                                       It’s the average over the test sample of the absolute
                                                                                                       differences between prediction and actual observation where
                                                                                                       all individual differences have equal weight.
                                                                                                       """])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                            ####
                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                            """,

                                                     """
                                                                                                       ###
                                                                                                       MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                                                                                                       """]),
                                     ('sentence 1', ["""
                            MAE is calculated as follows:
                            """,

                                                     """
                                                                                                       MAE is calculated using the following formula:
                                                                                                       """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Absolute Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Bias Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             If the absolute value is not taken (the signs of the errors are not removed),
                             the average error becomes the Mean Bias Error (MBE) and is
                             usually intended to measure average model bias.
                             MBE can convey useful information, but should be interpreted cautiously
                             because positive and negative errors will cancel out.
                             ####
                             """,

                                                     """
                                                                                                                      ####
                                                                                                                      If the absolute value is not taken (the signs of the errors are not removed),
                                                                                                                      the average error becomes the Mean Bias Error (MBE) and is
                                                                                                                      usually intended to measure average model bias.
                                                                                                                      MBE can convey useful information, but should be interpreted cautiously
                                                                                                                      because positive and negative errors will cancel out.
                                                                                                                      ####
                                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Squared Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The mean squared error is calculated as follows:
                             """,

                                                     """
                                                                                                      The mean squared error is calculated using the following formula:
                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Root Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                             It’s the square root of the average of squared differences between prediction and actual observation.
                             ###
                             """,

                                                     """
                                                                                                           ####
                                                                                                           RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                                                                                                           It’s the square root of the average of squared differences between prediction and actual observation.
                                                                                                           ###
                                                                                                           """]),
                                     ('sentence 1', ["""
                             RMSE does not necessarily increase with the variance of the errors.
                             RMSE is proportional to the variance of the error magnitude frequency distribution.
                             However, it is not necessarily proporional to the error variance.
                             """,

                                                     """
                                                                                                           RMSE does not necessarily increase with the variance of the errors.
                                                                                                           RMSE is proportional to the variance of the error magnitude frequency distribution.
                                                                                                           However, it is not necessarily proporional to the error variance.
                                                                                                           """]), ]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 1', ["""
                             RMSE can be appropriate when large errors should be penalized more than smaller errors.
                             The penalty of the error changes in a non-linear way.
                             """,

                                                     """
                                                                                                           ###
                                                                                                           If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate
                                                                                                           in some cases penalizing large errors more.
                                                                                                           If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate.
                                                                                                           ####
                                                                                                           """]),
                                     ('sentence 2', ["""
                             The root mean square error is calculated as follows:
                             """,

                                                     """
                                                                                                           The root mean square error is calculated using the following formula:
                                                                                                           """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Root Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [''])]
                                ))]
                            ),
                        })]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(
        os.path.join(os.path.abspath('../../LoParReportGenerator/'), 'HTML2'), 'ErrorMetrics.html'
    )

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean Absolute Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                MAE of a data set is the average distance between
                                each data value and the mean of a data set.
                                It describes the variation and dispersion in a data set.
                                """,

                                                     """
                                                                                                       MAE of a data set measures the average distance between
                                                                                                       each data value and the mean of a data set.
                                                                                                       It describes the variation and dispersion in a data set.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    It is the average magnitude
                                    of the errors in a set of predictions. Therefore, direction is not considered.
                                    """,

                                                     """
                                                                                                       Without their direction being considered, MAE averages the magnitude
                                                                                                       of the errors in a set of predictions.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    ####
                                    It’s the average over the test sample of the absolute
                                    differences between prediction and actual observation where
                                    all individual differences have equal weight.
                                    ####
                                    """,

                                                     """
                                                                                                       ####
                                                                                                       It’s the average over the test sample of the absolute
                                                                                                       differences between prediction and actual observation where
                                                                                                       all individual differences have equal weight.
                                                                                                       """])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                            ####
                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                            """,

                                                     """
                                                                                                       ###
                                                                                                       MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                                                                                                       """]),
                                     ('sentence 1', ["""
                            MAE is calculated as follows:
                            """,

                                                     """
                                                                                                       MAE is calculated using the following formula:
                                                                                                       """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Absolute Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Bias Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             If the absolute value is not taken (the signs of the errors are not removed),
                             the average error becomes the Mean Bias Error (MBE) and is
                             usually intended to measure average model bias.
                             MBE can convey useful information, but should be interpreted cautiously
                             because positive and negative errors will cancel out.
                             ####
                             """,

                                                     """
                                                                                                                      ####
                                                                                                                      If the absolute value is not taken (the signs of the errors are not removed),
                                                                                                                      the average error becomes the Mean Bias Error (MBE) and is
                                                                                                                      usually intended to measure average model bias.
                                                                                                                      MBE can convey useful information, but should be interpreted cautiously
                                                                                                                      because positive and negative errors will cancel out.
                                                                                                                      ####
                                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Squared Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The mean squared error is calculated as follows:
                             """,

                                                     """
                                                                                                      The mean squared error is calculated using the following formula:
                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Root Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                             It’s the square root of the average of squared differences between prediction and actual observation.
                             ###
                             """,

                                                     """
                                                                                                           ####
                                                                                                           RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                                                                                                           It’s the square root of the average of squared differences between prediction and actual observation.
                                                                                                           ###
                                                                                                           """]),
                                     ('sentence 1', ["""
                             RMSE does not necessarily increase with the variance of the errors.
                             RMSE is proportional to the variance of the error magnitude frequency distribution.
                             However, it is not necessarily proporional to the error variance.
                             """,

                                                     """
                                                                                                           RMSE does not necessarily increase with the variance of the errors.
                                                                                                           RMSE is proportional to the variance of the error magnitude frequency distribution.
                                                                                                           However, it is not necessarily proporional to the error variance.
                                                                                                           """]), ]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 1', ["""
                             RMSE can be appropriate when large errors should be penalized more than smaller errors.
                             The penalty of the error changes in a non-linear way.
                             """,

                                                     """
                                                                                                           ###
                                                                                                           If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate
                                                                                                           in some cases penalizing large errors more.
                                                                                                           If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate.
                                                                                                           ####
                                                                                                           """]),
                                     ('sentence 2', ["""
                             The root mean square error is calculated as follows:
                             """,

                                                     """
                                                                                                           The root mean square error is calculated using the following formula:
                                                                                                           """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Root Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [''])]
                                ))]
                            ),
                        })]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(
        os.path.join(os.path.abspath('../../generator/LoParReport/'), 'HTML2'), 'ErrorMetrics.html'
    )

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean Absolute Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                MAE of a data set is the average distance between
                                each data value and the mean of a data set.
                                It describes the variation and dispersion in a data set.
                                """,

                                                     """
                                                                                                       MAE of a data set measures the average distance between
                                                                                                       each data value and the mean of a data set.
                                                                                                       It describes the variation and dispersion in a data set.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    It is the average magnitude
                                    of the errors in a set of predictions. Therefore, direction is not considered.
                                    """,

                                                     """
                                                                                                       Without their direction being considered, MAE averages the magnitude
                                                                                                       of the errors in a set of predictions.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    ####
                                    It’s the average over the test sample of the absolute
                                    differences between prediction and actual observation where
                                    all individual differences have equal weight.
                                    ####
                                    """,

                                                     """
                                                                                                       ####
                                                                                                       It’s the average over the test sample of the absolute
                                                                                                       differences between prediction and actual observation where
                                                                                                       all individual differences have equal weight.
                                                                                                       """])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                            ####
                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                            """,

                                                     """
                                                                                                       ###
                                                                                                       MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                                                                                                       """]),
                                     ('sentence 1', ["""
                            MAE is calculated as follows:
                            """,

                                                     """
                                                                                                       MAE is calculated using the following formula:
                                                                                                       """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Absolute Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Bias Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             If the absolute value is not taken (the signs of the errors are not removed),
                             the average error becomes the Mean Bias Error (MBE) and is
                             usually intended to measure average model bias.
                             MBE can convey useful information, but should be interpreted cautiously
                             because positive and negative errors will cancel out.
                             ####
                             """,

                                                     """
                                                                                                                      ####
                                                                                                                      If the absolute value is not taken (the signs of the errors are not removed),
                                                                                                                      the average error becomes the Mean Bias Error (MBE) and is
                                                                                                                      usually intended to measure average model bias.
                                                                                                                      MBE can convey useful information, but should be interpreted cautiously
                                                                                                                      because positive and negative errors will cancel out.
                                                                                                                      ####
                                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Squared Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The mean squared error is calculated as follows:
                             """,

                                                     """
                                                                                                      The mean squared error is calculated using the following formula:
                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Root Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                             It’s the square root of the average of squared differences between prediction and actual observation.
                             ###
                             """,

                                                     """
                                                                                                           ####
                                                                                                           RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                                                                                                           It’s the square root of the average of squared differences between prediction and actual observation.
                                                                                                           ###
                                                                                                           """]),
                                     ('sentence 1', ["""
                             RMSE does not necessarily increase with the variance of the errors.
                             RMSE is proportional to the variance of the error magnitude frequency distribution.
                             However, it is not necessarily proporional to the error variance.
                             """,

                                                     """
                                                                                                           RMSE does not necessarily increase with the variance of the errors.
                                                                                                           RMSE is proportional to the variance of the error magnitude frequency distribution.
                                                                                                           However, it is not necessarily proporional to the error variance.
                                                                                                           """]), ]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 1', ["""
                             RMSE can be appropriate when large errors should be penalized more than smaller errors.
                             The penalty of the error changes in a non-linear way.
                             """,

                                                     """
                                                                                                           ###
                                                                                                           If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate
                                                                                                           in some cases penalizing large errors more.
                                                                                                           If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate.
                                                                                                           ####
                                                                                                           """]),
                                     ('sentence 2', ["""
                             The root mean square error is calculated as follows:
                             """,

                                                     """
                                                                                                           The root mean square error is calculated using the following formula:
                                                                                                           """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Root Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [''])]
                                ))]
                            ),
                        })]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter():
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../LoParReportGenerator1/'), 'HTML2'), 'ErrorMetrics.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean Absolute Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                MAE of a data set is the average distance between
                                each data value and the mean of a data set.
                                It describes the variation and dispersion in a data set.
                                """,

                                                     """
                                                                                                       MAE of a data set measures the average distance between
                                                                                                       each data value and the mean of a data set.
                                                                                                       It describes the variation and dispersion in a data set.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    It is the average magnitude
                                    of the errors in a set of predictions. Therefore, direction is not considered.
                                    """,

                                                     """
                                                                                                       Without their direction being considered, MAE averages the magnitude
                                                                                                       of the errors in a set of predictions.
                                                                                                       """]),
                                     ('sentence 1', ["""
                                    ####
                                    It’s the average over the test sample of the absolute
                                    differences between prediction and actual observation where
                                    all individual differences have equal weight.
                                    ####
                                    """,

                                                     """
                                                                                                       ####
                                                                                                       It’s the average over the test sample of the absolute
                                                                                                       differences between prediction and actual observation where
                                                                                                       all individual differences have equal weight.
                                                                                                       """])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                            ####
                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                            """,

                                                     """
                                                                                                       ###
                                                                                                       MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                                                                                                       """]),
                                     ('sentence 1', ["""
                            MAE is calculated as follows:
                            """,

                                                     """
                                                                                                       MAE is calculated using the following formula:
                                                                                                       """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Absolute Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Bias Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             If the absolute value is not taken (the signs of the errors are not removed),
                             the average error becomes the Mean Bias Error (MBE) and is
                             usually intended to measure average model bias.
                             MBE can convey useful information, but should be interpreted cautiously
                             because positive and negative errors will cancel out.
                             ####
                             """,

                                                     """
                                                                                                                      ####
                                                                                                                      If the absolute value is not taken (the signs of the errors are not removed),
                                                                                                                      the average error becomes the Mean Bias Error (MBE) and is
                                                                                                                      usually intended to measure average model bias.
                                                                                                                      MBE can convey useful information, but should be interpreted cautiously
                                                                                                                      because positive and negative errors will cancel out.
                                                                                                                      ####
                                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(Paths.FORMULAS_DIR, 'Mean Bias Error.png')),
                                     ('caption', collections.OrderedDict(
                                         [('paragraph 0', collections.OrderedDict(
                                             [('sentence 0', [''])]
                                         ))]
                                     ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Squared Error', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The mean squared error is calculated as follows:
                             """,

                                                     """
                                                                                                      The mean squared error is calculated using the following formula:
                                                                                                      """]), ]
                                )), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Root Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'hide':        False, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                             It’s the square root of the average of squared differences between prediction and actual observation.
                             ###
                             """,

                                                     """
                                                                                                           ####
                                                                                                           RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                                                                                                           It’s the square root of the average of squared differences between prediction and actual observation.
                                                                                                           ###
                                                                                                           """]),
                                     ('sentence 1', ["""
                             RMSE does not necessarily increase with the variance of the errors.
                             RMSE is proportional to the variance of the error magnitude frequency distribution.
                             However, it is not necessarily proporional to the error variance.
                             """,

                                                     """
                                                                                                           RMSE does not necessarily increase with the variance of the errors.
                                                                                                           RMSE is proportional to the variance of the error magnitude frequency distribution.
                                                                                                           However, it is not necessarily proporional to the error variance.
                                                                                                           """]), ]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 1', ["""
                             RMSE can be appropriate when large errors should be penalized more than smaller errors.
                             The penalty of the error changes in a non-linear way.
                             """,

                                                     """
                                                                                                           ###
                                                                                                           If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate
                                                                                                           in some cases penalizing large errors more.
                                                                                                           If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate.
                                                                                                           ####
                                                                                                           """]),
                                     ('sentence 2', ["""
                             The root mean square error is calculated as follows:
                             """,

                                                     """
                                                                                                           The root mean square error is calculated using the following formula:
                                                                                                           """])]
                                ))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', os.path.join(
                                        Paths.FORMULAS_DIR, 'Root Mean Squared Error.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'hide':        True, 'breakAfter': False, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [''])]
                                ))]
                            ),
                        })]
                )
            }),

        ]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)  # TODO
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter(data, dtypes=None):
    """
    :param vars: list of strings
    :param dtypes: list of strings
    :return:
    """

    imagesPath = os.path.join(os.path.abspath('/'), 'HTML2', 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = generate_all_plots(data, dtypes, imagesPath)
    regressionFit, regresssionError = regression
    plots = {var: ['Box plot', 'Distribution plot', 'Distribution Fit plot', 'Scatter plot',
                   'Violin plot'] if dtype == 'Numerical' else ['Bar plot'] for var, dtype in zip(vars, dtypes)}
    end = '. '
    captions = {}
    postText = {}

    def bar(figNum, numFigs=5):
        captions['Bar Plot'] = {}
        postText['Bar Plot'] = {}
        if 1 == 0:
            for var in vars:
                captions['Bar Plot'][var] = 'Figure {}: '.format(figNum)
                figNum += numFigs
            postText['Bar Plot'][var] = random.choice(
                ["""
                                                This is a bar plot0.""", """
                                                       Welcome to the bar plot0.
                                                       """]
            )
        return figNum

    def box(figNum, numFigs=5):
        captions['Box Plot'] = {}
        postText['Box Plot'] = {}

        for var in vars:
            captions['Box Plot'][var] = 'Figure {}: '.format(figNum)
            figNum += numFigs
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
                ['The minimum value found in <var>{0}</var> is <em>{1}</em>. '.format(var, minimum)]
            )

            # MAXIMUM
            boxText += random.choice(['The maximum is {}. '.format(maximum)])

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

            postText['Box Plot'][var] = boxText
        return figNum

    def distribution(figNum, numFigs=5):
        captions['Distribution Plot'] = {}
        postText['Distribution Plot'] = {}
        for var in vars:
            captions['Distribution Plot'][var] = 'Figure {}: '.format(figNum)
            figNum += numFigs
            if True:
                postText['Distribution Plot'][var] = random.choice(
                    ["""
                                                    This is a distribution plot0.""", """
                                                                    Welcome to the distribution plot0.
                                                                    """]
                )
        return figNum

    def distribution_fit(figNum, numFigs=5):
        captions['Distribution Fit Plot'] = {}
        postText['Distribution Fit Plot'] = {}
        for var in vars:
            keys = list(distFit[var].keys())
            captions['Distribution Fit Plot'][var] = 'Figure {}: '.format(figNum)
            figNum += numFigs
            distributionFitText = ''
            # Distribution Functions
            distributionFitText += random.choice(
                ['Performing a distribution fitting yielded the following distributions:',
                 'Of the distributions that were fit to the data, the following were the best fitting:',
                 'This distribution of this data is best approximated by the following distributions:']
            )
            distributionFitText += random.choice(
                [r'$$\textbf{' + keys[0] + ':}$$' + '$${}$$'.format(
                    distFit[var][keys[0]]
                ) + r'$$\textbf{' + keys[1] + ':}$$' + '$${}$$'.format(distFit[var][keys[1]])]
            )
            # TODO sentence about goodness-of-fit
            postText['Distribution Fit Plot'][var] = distributionFitText
        return figNum

    def scatter(figNum, numFigs=5):
        captions['Scatter Plot'] = {}
        postText['Scatter Plot'] = {}

        for var in vars:
            fit = s_round(regressionFit[var])
            mse = s_round(regressionError[var]['mse'])
            R2 = s_round(regressionError[var]['R2'])
            rmse = s_round(regressionError[var]['rmse'])
            mae = s_round(regressionError[var]['mae'])
            captions['Scatter Plot'][var] = 'Figure {}: '.format(figNum)
            figNum += numFigs
            scatterText = ''
            scatterText += random.choice(
                ['Using curve fitting procedures, '
                 '<var>{0}</var> was found to be best described by $${1}$$. '.format(var, fit),
                 'Through a Regression6 analysis, the function'
                 '$${1}$$ was found to best fit <var>{0}</var>. '.format(var, fit)]
            )
            scatterText += random.choice(['The <var>mean squared error</var> of this fit is <em>{}</em>. '.format(mse)])
            scatterText += random.choice(['The <var>R squared</var> of this fit is <em>{}</em>. '.format(R2)])
            scatterText += random.choice(
                ['The <var>root mean squared error</var> of this fit is <em>{}</em>. '.format(rmse)]
            )
            scatterText += random.choice(
                ['The <var>mean absolute error</var> of this fit is <em>{}</em>. '.format(mae)]
            )

            postText['Scatter Plot'][var] = scatterText
        return figNum

    def violin(figNum, numFigs=5):
        captions['Violin Plot'] = {}
        postText['Violin Plot'] = {}
        for var in vars:
            captions['Violin Plot'][var] = 'Figure {}: '.format(figNum)
            figNum += numFigs
            # CHECK STATISTICS
            mean = numStats[var]['Mean']
            median = numStats[var]['Median']
            mode = numStats[var]['Mode']
            std = numStats[var]['Standard\nDeviation']
            skew = numStats[var]['Skew']
            kurtosis = numStats[var]['Kurtosis']
            x = s_round(mean - median)
            c0 = s_round(mean / median)
            c1 = s_round((abs(x) / mean) * 100)

            violinText = ''

            # MEAN ----------------------------------------------
            meanSentence = random.choice(
                ['The distribution of <var>' + var + '</var> has a <em>mean</em> of ' + str(mean),

                 '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
            ) + end

            # MEDIAN --------------------------------------------
            medianSentence = random.choice(['The <em>median</em> is ' + str(median)]) + end
            if x > 0:
                """Mean is greater than median"""
                meanMedianComparison = 'The <em>mean</em> is ' + str(c0) + ' times, or ' + str(
                    c1
                ) + '% greater than the <em>median</em>. ' + end
            else:
                """Median is greater than mean"""
                meanMedianComparison = 'The <em>mean</em> is ' + str(s_round(1 / c0)) + ' times, or ' + str(
                    c1
                ) + '% smaller than the <em>median</em>. ' + end

            # MODE ----------------------------------------------
            modeSentence = ''

            # SKEW ----------------------------------------------
            skewText = random.choice(
                ['The <em>skew</em> of the distribution is ' + str(
                    skew
                ) + ', which indicates that the data is concentrated ']
            )

            if abs(skew) <= 0.5:
                """fairly symmetrical"""
                skewText += random.choice(['slightly']) + ' to the '
            elif abs(skew) <= 1:
                """moderately skew"""
                skewText += random.choice(['moderately']) + ' to the '
            else:
                """highly skewed"""
                skewText += random.choice(['highly']) + ' to the '

            if skew > 0:
                skewText += ' left with a longer tail to the right. '
            elif skew < 0:
                skewText += ' right with a longer tail to the left. '
            else:
                concentration = 'center'
                tail = ' with equally long tails on both sides.' \
                       'Thus this distribution is perfectly symmetrical'

            # KURTOSIS ------------------------------------------
            kurtosisText = random.choice(['The <em>kurtosis</em> of ' + str(kurtosis) + ' means that the data is '])
            if kurtosis < 0:
                kurtosisText += ' a very spread out' \
                                ' <em>platykurtic</em> distribution. Therefore, the tails of the distribution' \
                                ' are relatively long, slender, and more prone to contain outliers'
            elif kurtosis > 0:
                kurtosisText += ' a tightly concentrated' \
                                ' <em>leptokurtic</em> distribution. Therefore, the tails of the distribution' \
                                ' are relatively short, broad, and less prone to contain outliers'
            else:
                kurtosisText += ' a moderately spread out' \
                                ' <em>mesokurtic</em> distribution.'

            violinText = meanSentence + ' ' + medianSentence + ' ' + meanMedianComparison + ' ' + modeSentence + ' ' + skewText + ' ' + kurtosisText

            postText['Violin Plot'][var] = violinText
        return figNum

    # bar(1)
    box(1)
    distribution(2)
    distribution_fit(3)
    scatter(4)
    violin(5)
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('/'), 'HTML2'), 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[:-1], {
                            'hide':        False, 'breakAfter': True, 'displayTitle': True,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure ' + str(i + 1), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(Paths.PLOTS_DIR, plot.replace(' ', '')[:-5]), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [captions[plot[:-1]][var]])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0',
                                  collections.OrderedDict([('sentence 0', [postText[plot[:-1]][var]]), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for var in vars]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter(data, dtypes=None):
    """
    :param vars: list of strings
    :param dtypes: list of strings
    :return:
    """

    imagesPath = os.path.join(os.path.join(os.path.abspath('../LoParReportGenerator1/'), 'HTML2'), 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = generate_all_plots(data, dtypes, imagesPath)
    plots = {var: ['Box plot', 'Distribution plot', 'Distribution Fit plot', 'Scatter plot',
                   'Violin plot'] if dtype == 'Numerical' else ['Bar plot'] for var, dtype in zip(vars, dtypes)}
    end = '. '
    captions = {}
    postText = {}

    def bar():
        captions['Bar Plot'] = {}
        postText['Bar Plot'] = {}
        if 1 == 0:
            for var in vars:
                captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
            postText['Bar Plot'][var] = random.choice(
                ["""
                                                This is a bar plot0.""", """
                                                       Welcome to the bar plot0.
                                                       """]
            )

    def box():
        captions['Box Plot'] = {}
        postText['Box Plot'] = {}

        for var in vars:
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
                ['The minimum value found in <var>{0}</var> is <em>{1}</em>. '.format(var, minimum)]
            )

            # MAXIMUM
            boxText += random.choice(['The maximum is {}. '.format(maximum)])

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

            postText['Box Plot'][var] = boxText

    def distribution():
        captions['Distribution Plot'] = {}
        postText['Distribution Plot'] = {}
        for var in vars:
            captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
            if True:
                postText['Distribution Plot'][var] = random.choice(
                    ["""
                                                    This is a distribution plot0.""", """
                                                                    Welcome to the distribution plot0.
                                                                    """]
                )

    def distribution_fit():
        captions['Distribution Fit Plot'] = {}
        postText['Distribution Fit Plot'] = {}
        for var in vars:
            keys = list(distFit[var].keys())
            func1 = distFit[var][keys[0]][0]
            func2 = distFit[var][keys[1]][0]
            error1 = s_round(distFit[var][keys[0]][1])
            error2 = s_round(distFit[var][keys[1]][1])
            captions['Distribution Fit Plot'][var] = 'Distribution fit of <var>{0}</var>.'.format(var)
            distributionFitText = ''
            # Distribution functions
            distributionFitText += random.choice(
                ['Performing a distribution fitting yielded the following best fit distributions:',
                 'Of the distributions that were fit to the data, the following were the best fitting:',
                 'This distribution of this data is best approximated by the following distributions:',
                 'The results of curve fitting analysis indicate that the following functions best fit the data:']
            )
            distributionFitText += random.choice(
                [r'$$\textbf{' + keys[0] + ':}$$' + '$${}$$'.format(
                    func1
                ) + r'$$\textbf{' + keys[1] + ':}$$' + '$${}$$'.format(func2)]
            )

            # Goodness of fit
            distributionFitText += random.choice(
                ['With <var>{0}</var> having a <em>sum of squared errors</em> of {1},'
                 'and <var>{2}</var> having a <em>sum of squared errors</em> of {3}. '.format(
                    keys[0], error1, keys[1], error2
                ), 'With <var>{0}</var> and <var>{1}</var> having a <em>sum of squared errors</em> of '
                   '{2} and {3}, respectively. '.format(keys[0], keys[1], error1, error2), ]
            )

            postText['Distribution Fit Plot'][var] = distributionFitText

    def scatter():
        captions['Scatter Plot'] = {}
        postText['Scatter Plot'] = {}

        for var in vars:
            fit = s_round(regression[var][0])
            errors = regression[var][1]
            mse = s_round(errors['mse'])
            R2 = s_round(errors['R2'])
            rmse = s_round(errors['rmse'])
            mae = s_round(errors['mae'])
            sor = s_round(errors['sum of residuals'])
            chi2 = s_round(errors['chi2'])
            reducedChi2 = s_round(errors['reduced chi2'])
            standardError = s_round(errors['standard error'])
            captions['Scatter Plot'][var] = 'Scatter plot0 and Regression6 fit of <var>{0}</var>.'.format(var)
            scatterText = ''
            scatterText += random.choice(
                ['Using curve fitting procedures, '
                 '<var>{0}</var> was found to be best described by $${1}$$'.format(var, fit),
                 'Through a Regression6 analysis, the function'
                 '$${1}$$ was found to best fit <var>{0}</var>. '.format(var, fit)]
            )
            scatterText += random.choice(['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)])
            scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
            scatterText += random.choice(
                ['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)]
            )
            scatterText += random.choice(['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)])
            scatterText += random.choice(
                [
                    'The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
                    'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
            )
            scatterText += random.choice(
                ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
                 'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(
                    standardError, sor
                )]
            )

            postText['Scatter Plot'][var] = scatterText

    def violin():
        captions['Violin Plot'] = {}
        postText['Violin Plot'] = {}
        for var in vars:
            captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
            # CHECK STATISTICS
            mean = numStats[var]['Mean']
            median = numStats[var]['Median']
            mode = numStats[var]['Mode']
            std = numStats[var]['Standard\nDeviation']
            skew = numStats[var]['Skew']
            kurtosis = numStats[var]['Kurtosis']
            x = s_round(mean - median)
            c0 = s_round(mean / median)
            c1 = s_round((abs(x) / mean) * 100)

            violinText = ''

            # MEAN ----------------------------------------------
            meanSentence = random.choice(
                ['The distribution of <var>' + var + '</var> has a <em>mean</em> of ' + str(mean),

                 '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
            ) + end

            # MEDIAN --------------------------------------------
            medianSentence = random.choice(['The <em>median</em> is ' + str(median)]) + end
            if x > 0:
                """Mean is greater than median"""
                meanMedianComparison = 'The <em>mean</em> is ' + str(c0) + ' times, or ' + str(
                    c1
                ) + '% greater than the <em>median</em>. ' + end
            else:
                """Median is greater than mean"""
                meanMedianComparison = 'The <em>mean</em> is ' + str(s_round(1 / c0)) + ' times, or ' + str(
                    c1
                ) + '% smaller than the <em>median</em>. ' + end

            # MODE ----------------------------------------------
            modeSentence = ''

            # SKEW ----------------------------------------------
            skewText = random.choice(
                ['The <em>skew</em> of the distribution is ' + str(
                    skew
                ) + ', which indicates that the data is concentrated ']
            )

            if abs(skew) <= 0.5:
                """fairly symmetrical"""
                skewText += random.choice(['slightly'])
            elif abs(skew) <= 1:
                """moderately skew"""
                skewText += random.choice(['moderately'])
            else:
                """highly skewed"""
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
                                ' are relatively long, slender, and more prone to contain outliers'
            elif kurtosis > 0:
                kurtosisText += ' a tightly concentrated' \
                                ' <em>leptokurtic</em> distribution. Therefore, the tails of the distribution' \
                                ' are relatively short, broad, and less prone to contain outliers'
            else:
                kurtosisText += ' a moderately spread out' \
                                ' <em>mesokurtic</em> distribution.'

            violinText = meanSentence + ' ' + medianSentence + ' ' + meanMedianComparison + ' ' + modeSentence + ' ' + skewText + ' ' + kurtosisText

            postText['Violin Plot'][var] = violinText

    bar()
    box()
    distribution()
    distribution_fit()
    scatter()
    violin()

    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(os.path.abspath('../LoParReportGenerator1/'), 'HTML2'), 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[:-1], {
                            'hide':        False, 'breakAfter': True, 'displayTitle': False,
                            'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ), 'figures':  collections.OrderedDict(
                                [('Figure ' + str(i + 1), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(Paths.PLOTS_DIR, plot.replace(' ', '')[:-5]), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText': collections.OrderedDict(
                                [('paragraph 0',
                                  collections.OrderedDict([('sentence 0', [postText[plot[:-1]][var]]), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    chapterAbstract = Generator.generate_textStruct(chapterAbstract)
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, heavyBreak=False)


def write_chapter(path=None, templateDir=None):
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(path, 'HTML2'), 'DescriptiveStatistics.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Descriptive Statistics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""

                """,

                                    """

                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}
    # 'displayTitle' determines
    # whether the title is displayed or not.
    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.
    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}
    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}
    # 'displayTitle' determines whether
    # the subSection title is displayed.
    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The Mean of your data set is also known as the average
                                your data set.
                                """,

                                                     """
                                                                                                The Average of your data set is what is called the Mean
                                                                                                or central value of the data set.
                                                                                              """,

                                                     """
                                                                                                The Mean is the average of all data points within the
                                                                                                data set and is sometime called the Arithmetic Mean.
                                                                                              """,

                                                     """
                                                                                                One way to find the Central Tendency of your data is
                                                                                                to find the Mean (average) of your data set.
                                                                                              """, """
                                       The Mean is the Average of your data set
                                     """, """
                                       The average of all the data points within the data
                                       set is called the Mean of the the data set.
                                     """]), ('sentence 1', ["""
                                The average is calculated by first adding up all
                                the data points within your data set. Take the
                                sum and divide it by the total number of data points
                                within your set.
                              """,

                                                            """
                                                              You must first add up the data points within your data
                                                              set and then divide by the total number of data points.
                                                            """,

                                                            """
                                                              The mean is calculated by adding up all the data points
                                                              and dividing by the total number of data points.
                                                            """,

                                                            """
                                                              To get the Mean you must add up all the data points. take
                                                              the sum and divide by the total number of data points.
                                                            """,

                                                            """
                                                              The Mean is the summation of the data set points divided
                                                              by the number of dataset points.
                                                            """,

                                                            """
                                                              To calculate the Average (Mean), add together all the
                                                              data points within the data set, and then divide the
                                                              sum by the total number of data points within the data set.
                                                            """,

                                                            """
                                                              Add all the values together and divide them by the number
                                                              of data points within the data set.
                                                            """]), ('sentence 1', ["""
                                Often with larger data sets it may be easier to define
                                the data set by a single number, known as the Central Tendency.
                                Calculating the Mean is one way to find this number.
                              """,

                                                                                   """
                                                                                     When dealing with a larger data set it may be easier to use a singular
                                                                                     value to define it, this value is called the Central Tendency. Calculating
                                                                                     the mean is one of the ways to find the Central Tendency.
                                                                                   """,

                                                                                   """
                                                                                     This gives you a central number to define your data
                                                                                   """, ])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The following formula is used for calculating Mean.
                            """,

                                                     """
                                                                                                         """]),
                                     ('sentence 1', ["""
                            Mean is calculated as follows:
                            """,

                                                     """
                                                                                                         Mean is calculated using the following formula:
                                                                                                         """])]
                                ))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Median', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                 The Median is the midpoint of the data set. It is the
                                 point that separates the lower half of your data from the top half.
                              """,

                                                     """
                                                                                                               The midpoint of the data set is known as the Median. The Median is
                                                                                                               used to represent the center of your data set.
                                                                                                             """,

                                                     """
                                                                                                               Your data can be split into two sets, a lower half and a top half.
                                                                                                               The midpoint of the set is known as the Median
                                                                                                             """,

                                                     """
                                                                                                               The Median is the data point lying at the midpoint of the data set.
                                                                                                             """,

                                                     """
                                                                                                               The Median is the center value within a data set.
                                                                                                             """,

                                                     """
                                                                                                               The median is the middle value of the data set
                                                                                                             """,

                                                     """
                                                                                                               The median is the middle most value of data set.
                                                                                                             """]),
                                     ('sentence 1', ["""
                                If you want to find the Median of your data set,
                                set all the data points within the data set in numerical order.
                             """,

                                                     """
                                                                                                                   In order to get the Median set the data points within the data set
                                                                                                                   in numerical order.
                                                                                                                """,

                                                     """
                                                                                                                  You can get the Median of a data set by setting out all the
                                                                                                                  data points in order.
                                                                                                                """,

                                                     """
                                                                                                                  The Median is found by locating the midpoint within the data set.
                                                                                                                """,

                                                     """
                                                                                                                  Order the values within the data set in numerical order.
                                                                                                                  Look for the center value.
                                                                                                                """,

                                                     """
                                                                                                                  Lay out the data in numerical order.
                                                                                                                """,

                                                     """
                                                                                                                  Lay out the data set from least to greatest and locate
                                                                                                                  the middle value.
                                                                                                                """

                                                     ]), ('sentence 1', ["""
                                If your data set does not have a midpoint due to there being an
                                even number set of data points. Find the mean of the two midpoints.
                             """,

                                                                         """
                                                                                                                                      When the data set has an even number of data points the Mean of the
                                                                                                                                      two most center points will be taken and that will be known as the
                                                                                                                                      median of the data set.
                                                                                                                                    """,

                                                                         """
                                                                                                                                      If the midpoint of your data set cannot be located due to the
                                                                                                                                      data having an odd number of data points. Then the two midpoints of
                                                                                                                                      the data set will be taken and the Mean of the two points will be
                                                                                                                                      found and become the median.
                                                                                                                                    """,

                                                                         """
                                                                                                                                      If the data set does not have a center point. Find the two most midpoints
                                                                                                                                      within the data set and find their mean. That Mean is the Median of the
                                                                                                                                      data set.
                                                                                                                                    """,

                                                                         """
                                                                                                                                      If the data has an even set of numbers and no center point then calculate
                                                                                                                                      the mean of the two center points.
                                                                                                                                    """,

                                                                         """
                                                                                                                                      If the data set has an odd number of data points the midpoint is the median.
                                                                                                                                      If the number of data points within the data set equates to an even number
                                                                                                                                      of data points. Then the Mean of the two mid points will be the median of
                                                                                                                                      the data set.
                                                                                                                                    """])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mode', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The Mode of the dataset is the most reccuring data
                                point through out the data set
                              """,

                                                     """
                                                                                                The Value or frequency to appear within the data set
                                                                                                most often is referred to as the Mode
                                                                                              """,

                                                     """
                                                                                                The Mode is the value or data point that appears most
                                                                                                often within the data set
                                                                                              """,

                                                     """
                                                                                                When a data set contains a value that occurs more than
                                                                                                once it is known as the Mode. This is the Value most likely
                                                                                                to be picked fro the data set
                                                                                              """,

                                                     """
                                                                                              The value that appears most often is the Mode
                                                                                              """,

                                                     """
                                                                                              The value that appears most frequently within the data set
                                                                                              """]), ('sentence 1', ["""
                                One you have found the Median look for the value that
                                is most recurring this the Mode
                             """,

                                                                                                                     """
                                                                                                                             It is easiest to find the Mode once you have the Median.
                                                                                                                           """,

                                                                                                                     """
                                                                                                                             After you have found the Median of the data set it is
                                                                                                                             much easier to locate which value occurs the most with
                                                                                                                             the data points laid out.
                                                                                                                           """,

                                                                                                                     """
                                                                                                                             To Calculate the mode align the data points is numerical
                                                                                                                             order and find the data point the occurs the most often.
                                                                                                                           """,

                                                                                                                     """
                                                                                                                             The Mode is easiest to locate after the median is located.
                                                                                                                             Once you have the data points laid out locate the value
                                                                                                                             that recurrs the most often
                                                                                                                           """,

                                                                                                                     """
                                                                                                                             Put the data points from least to greatest and count
                                                                                                                             the value that recurrs the most
                                                                                                                           """, ]),
                                     ('sentence 1', ["""
                                A dataset can contain more contain more than one mode.
                                If there are only recurring values the data set is Bimodal.
                             """,

                                                     """
                                                                        If the dataset has two recurring values then it is Bimodal
                                                                        """,

                                                     """
                                                                        Often a data set may contain more than one recurring value.
                                                                        In such cases the data set is referred to as bimodal
                                                                        """,

                                                     """
                                                                        A bimodal data set is a set that contains two Modal Values
                                                                        """,

                                                     """
                                                                        A data set containing two modal values is a Bimodal data set
                                                                        """,

                                                     """
                                                                        If two values recurr throughout the data an equal amount of
                                                                        times the data set is Bimodal
                                                                        """]), ('sentence 2', ["""
                                If the dataset has three recurring values then the data set is tri modal
                             """,

                                                                                               """
                                                                          When the data set has three recurring values it is Trimodal
                                                                        """,

                                                                                               """
                                                                          If the data set contains three recurring values. The data set is trimodal
                                                                        """,

                                                                                               """
                                                                          A trimodal data set is a data set that contains three modal values
                                                                        """,

                                                                                               """
                                                                          A data set containing three modal values is a Trimodal
                                                                        """,

                                                                                               """
                                                                          If three values recur throughout the data an equal amount of times
                                                                          the dataset is trimodal
                                                                        """]), ('sentence 3', ["""
                              If the dataset has more than three recurring values then the data set is multi modal
                           """,

                                                                                               """
                                                                                    When the data set has above three recurring values. The data set is multi modal
                                                                                   """,

                                                                                               """
                                                                                    In larger data sets you may see multiple recurring data sets,
                                                                                    In such cases the data set is multi modal
                                                                                   """,

                                                                                               """
                                                                                    A multi modal set is a dataset that contains multiple recurring values
                                                                                   """,

                                                                                               """
                                                                                    A data set containing multiple values is Multimodal
                                                                                   """,

                                                                                               """
                                                                                    If multiple values recur throughout the dataset the set is multimodal
                                                                                   """])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Skew', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                    The Skew of a data set is the measure of the lack of
                                    symmetry within the data. The Skewness of the data
                                    can be either positive, negative, or undefined
                                  """,

                                                     """
                                                                                                Skewness describes the symmetry of a distribution.
                                                                                                Or lack there of symmetry. If the data has a tail end
                                                                                                distribution then the data is skewed.
                                                                                              """,

                                                     """
                                                                                                The Distribution of a data set is the Skewness of the data set.
                                                                                                This defines whether the data is symmetrical or not
                                                                                              """,

                                                     """
                                                                                                The skew defines the distribution of the data set.
                                                                                              """,

                                                     """
                                                                                                Skewness is when the curve of the data is distorted to
                                                                                                either the left or the right
                                                                                              """,

                                                     """

                                                                                              """]), ('sentence 1', ["""
                                    If the data is skewed to the right this is known as a positive
                                    distribution. In a positive distribution the mean usually greater
                                    than the median, both are to the right of the Mode.
                                  """,

                                                                                                                     """
                                                                                                                             If the tail of the data extends to the right then the data
                                                                                                                             is positively skewed. In a positively skewed distribution
                                                                                                                             the Mean is usually to the right of the median and both are
                                                                                                                             to the right of the Mode.
                                                                                                                           """,

                                                                                                                     """
                                                                                                                             When the tail of the data lays to the right of the curve.
                                                                                                                             The data has a positive skew. In a positive skew the Mean is
                                                                                                                             greater than the median and both are to the right of the mode.
                                                                                                                           """,

                                                                                                                     """
                                                                                                                             If the right side of the data has a tail then the data is positively
                                                                                                                             skewed. The mean and the median lie to the right of the mode.
                                                                                                                           """,

                                                                                                                     """
                                                                                                                             A curve aligned to the right extending a tail to the left is a
                                                                                                                             negatively skewed curve.
                                                                                                                           """,

                                                                                                                     """

                                                                                                                           """, ]),
                                     ('sentence 1', ["""
                                    If the data is skewed to the left this is known as a negative distribution.
                                    In a negative distribution the Mean is less than the median and both are
                                    less than the mode.
                                  """,

                                                     """
                                                                          If the tail of the data extends to the left then the data is negatively skewed.
                                                                          In a negative distribution the Mean is to the left of the median and both are
                                                                          to the left of the mode.
                                                                        """,

                                                     """
                                                                          When the tail of the data lays to the left of the curve the mean
                                                                          is less than the median and both are to the left of the curve.
                                                                        """,

                                                     """
                                                                          If the left side of the data has a tail end.
                                                                          The data is negatively skewed.
                                                                          The mean is less than the median and the Mode.
                                                                        """,

                                                     """
                                                                          A curve aligned to the left extending a tail to the
                                                                          right is a positively skewed curve.
                                                                        """,

                                                     """

                                                                        """]), ('sentence 2', ["""
                                    The distribution of the data is symmetrical when the data lacks skewness.
                                  """,

                                                                                               """
                                                                          When the data does not contain a tail it is a bell curve.
                                                                          In a bell curve the data is symmetrical.
                                                                        """,

                                                                                               """
                                                                          If there is no extending tail in the data the data is a Bell Curve.
                                                                          A symmetrical data set
                                                                        """,

                                                                                               """
                                                                          A data set is symmetrical if it looks the same on the
                                                                          left and the right side.
                                                                        """,

                                                                                               """
                                                                          A curve in the center of the data is a bell curve,
                                                                          a symmetrical curve.
                                                                        """,

                                                                                               """

                                                                        """]), ('sentence 3', ["""

                                  """,

                                                                                               """

                                                                                   """,

                                                                                               """

                                                                                   """,

                                                                                               """

                                                                                   """,

                                                                                               """

                                                                                   """,

                                                                                               """

                                                                                   """])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                ),
            }),

            ('Kurtosis', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                  Kurtosis is the measure of sharpnesss in the peak of the distribution.
                                  If it is a positive, negative, or normal curve.
                               """,

                                                     """
                                                                                                                 The measure of the shape of the curve.
                                                                                                                 Kurtosis measures if it is flat, normal, or peaked
                                                                                                               """,

                                                     """
                                                                                                                 Kurtosis is the measure of the shape of the distribution.
                                                                                                                 A measure of Kurtosis can either be a 0, a negative, or a positive number
                                                                                                               """,

                                                     """
                                                                                                                 Kurtosis measures how flat or peaked out data distribution is.
                                                                                                               """,

                                                     """
                                                                                                                 Kurtosis refers to the peakedness or flatness of the distribution.
                                                                                                               """,

                                                     """

                                                                                                               """]),
                                     ('sentence 1', ["""
                                   When the data does not have a peak distribution but
                                   resembles a normal bell curve this is called Mesokurtic.
                               """,

                                                     """
                                                                                                                  When the curve has the bell curve of a normal distribution
                                                                                                                  then it is a Mesokurtic Distribution
                                                                                                                """,

                                                     """
                                                                                                                  When the calculation of Kurtosis is that of a normal distribution
                                                                                                                  then the distribution is that of a Mesokurtic
                                                                                                                """,

                                                     """
                                                                                                                  When the measure of Kurtosis is a positive distribution it will
                                                                                                                  have a extended peak. This is known as Mesokurtic.
                                                                                                                """,

                                                     """
                                                                                                                  When the distribution has a sharp peak this is a positive
                                                                                                                  kurtosis known as a mesokurtic distribution
                                                                                                                """,

                                                     """

                                                                                                                """, ]),
                                     ('sentence 1', ["""
                                   A negative kurtosis is a Mesokurtic distribution.
                                   This will have closer to what looks like a flat distribution.
                               """,

                                                     """
                                                                                       When the curve is flatter than a normal bell distribution
                                                                                       this is called a Platykurtic distribution
                                                                                     """,

                                                     """
                                                                                       When the calculation of Kurtosis is a negative number this
                                                                                       results in a Platykurtic distribution. A peak that is flatter
                                                                                       than a normal distribution.
                                                                                     """,

                                                     """
                                                                                       When the measure of Kurtosis has a negative distribution it
                                                                                       will have a flatter lower curve this curve is known as Platykurtic.
                                                                                     """,

                                                     """

                                                                                     """,

                                                     """

                                                                                     """]), ('sentence 2', ["""
                                   A positive kurtosis is called Leptokurtic.
                                   In a Leptokurtic distribution the data has a sharp peak
                               """,

                                                                                                            """
                                                                                       When the curve is taller and skinnier than a normal
                                                                                       distribution it has a positive Kurtosis also known
                                                                                       as a Letokurtic Distribution
                                                                                     """,

                                                                                                            """
                                                                                       When the calculation of Kurtosis is a positive number
                                                                                       this results in a Leptokurtic distribution, or a sharp peak.
                                                                                     """,

                                                                                                            """
                                                                                       When the measure of Kutosis has a regular distribution or a
                                                                                       distribution of 0 it will resemble the normal bell ccurve,
                                                                                       also known as a Leptokurtic distribution.
                                                                                     """,

                                                                                                            """

                                                                                     """,

                                                                                                            """

                                                                                     """]), ('sentence 3', ["""

                               """,

                                                                                                            """

                                                                                   """,

                                                                                                            """

                                                                                   """,

                                                                                                            """

                                                                                   """,

                                                                                                            """

                                                                                   """,

                                                                                                            """

                                                                                   """])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            })]
    )
    html = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir=templateDir)


def write_chapter(path=None, jobName=None, templateDir=None):
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(path, jobName), 'DescriptiveStatistics.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Descriptive Statistics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""

                """,

                                    """

                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The Mean of your data set is also known as the average
                                your data set.
                                """,

                                                     """
                                                                                               The Average of your data set is what is called the Mean
                                                                                               or central value of the data set.
                                                                                             """,

                                                     """
                                                                                               The Mean is the average of all data points within the
                                                                                               data set and is sometime called the Arithmetic Mean.
                                                                                             """,

                                                     """
                                                                                               One way to find the Central Tendency of your data is
                                                                                               to find the Mean (average) of your data set.
                                                                                             """, """
                                The Mean is the Average of your data set
                              """, """
                                The average of all the data points within the data
                                set is called the Mean of the the data set.
                              """]), ('sentence 1', ["""
                                The average is calculated by first adding up all
                                the data points within your data set. Take the
                                sum and divide it by the total number of data points
                                within your set.
                              """,

                                                     """
                                                       You must first add up the data points within your data
                                                       set and then divide by the total number of data points.
                                                     """,

                                                     """
                                                       The mean is calculated by adding up all the data points
                                                       and dividing by the total number of data points.
                                                     """,

                                                     """
                                                       To get the Mean you must add up all the data points. take
                                                       the sum and divide by the total number of data points.
                                                     """,

                                                     """
                                                       The Mean is the summation of the data set points divided
                                                       by the number of dataset points.
                                                     """,

                                                     """
                                                       To calculate the Average (Mean), add together all the
                                                       data points within the data set, and then divide the
                                                       sum by the total number of data points within the data set.
                                                     """,

                                                     """
                                                       Add all the values together and divide them by the number
                                                       of data points within the data set.
                                                     """]), ('sentence 1', ["""
                                Often with larger data sets it may be easier to define
                                the data set by a single number, known as the Central Tendency.
                                Calculating the Mean is one way to find this number.
                              """,

                                                                            """
                                                                              When dealing with a larger data set it may be easier to use a singular
                                                                              value to define it, this value is called the Central Tendency. Calculating
                                                                              the mean is one of the ways to find the Central Tendency.
                                                                            """,

                                                                            """
                                                                              This gives you a central number to define your data
                                                                            """, ])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The following formula is used for calculating Mean.
                            """,

                                                     """
                                                                                              """]), ('sentence 1', ["""
                            Mean is calculated as follows:
                            """,

                                                                                                                     """
                                                                                                                     Mean is calculated using the following formula:
                                                                                                                     """]),
                                     ('sentence 1', [ExpressionGenerator.mean_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Median', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                 The Median is the midpoint of the data set. It is the
                                 point that separates the lower half of your data from the top half.
                              """,

                                                     """
                                                                                                 The midpoint of the data set is known as the Median. The Median is
                                                                                                 used to represent the center of your data set.
                                                                                               """,

                                                     """
                                                                                                 Your data can be split into two sets, a lower half and a top half.
                                                                                                 The midpoint of the set is known as the Median
                                                                                               """,

                                                     """
                                                                                                 The Median is the data point lying at the midpoint of the data set.
                                                                                               """,

                                                     """
                                                                                                 The Median is the center value within a data set.
                                                                                               """,

                                                     """
                                                                                                 The median is the middle value of the data set
                                                                                               """,

                                                     """
                                                                                                 The median is the middle most value of data set.
                                                                                               """]), ('sentence 1', ["""
                                If you want to find the Median of your data set,
                                set all the data points within the data set in numerical order.
                             """,

                                                                                                                      """
                                                                                                                         In order to get the Median set the data points within the data set
                                                                                                                         in numerical order.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        You can get the Median of a data set by setting out all the
                                                                                                                        data points in order.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        The Median is found by locating the midpoint within the data set.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        Order the values within the data set in numerical order.
                                                                                                                        Look for the center value.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        Lay out the data in numerical order.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        Lay out the data set from least to greatest and locate
                                                                                                                        the middle value.
                                                                                                                      """

                                                                                                                      ]),
                                     ('sentence 1', ["""
                                If your data set does not have a midpoint due to there being an
                                even number set of data points. Find the mean of the two midpoints.
                             """,

                                                     """
                                                                                                 When the data set has an even number of data points the Mean of the
                                                                                                 two most center points will be taken and that will be known as the
                                                                                                 median of the data set.
                                                                                               """,

                                                     """
                                                                                                 If the midpoint of your data set cannot be located due to the
                                                                                                 data having an odd number of data points. Then the two midpoints of
                                                                                                 the data set will be taken and the Mean of the two points will be
                                                                                                 found and become the median.
                                                                                               """,

                                                     """
                                                                                                 If the data set does not have a center point. Find the two most midpoints
                                                                                                 within the data set and find their mean. That Mean is the Median of the
                                                                                                 data set.
                                                                                               """,

                                                     """
                                                                                                 If the data has an even set of numbers and no center point then calculate
                                                                                                 the mean of the two center points.
                                                                                               """,

                                                     """
                                                                                                 If the data set has an odd number of data points the midpoint is the median.
                                                                                                 If the number of data points within the data set equates to an even number
                                                                                                 of data points. Then the Mean of the two mid points will be the median of
                                                                                                 the data set.
                                                                                               """]),
                                     ('sentence 2', [ExpressionGenerator.median_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mode', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The Mode of the dataset is the most reccuring data
                                point through out the data set
                              """,

                                                     """
                                                                                               The Value or frequency to appear within the data set
                                                                                               most often is referred to as the Mode
                                                                                             """,

                                                     """
                                                                                               The Mode is the value or data point that appears most
                                                                                               often within the data set
                                                                                             """,

                                                     """
                                                                                               When a data set contains a value that occurs more than
                                                                                               once it is known as the Mode. This is the Value most likely
                                                                                               to be picked fro the data set
                                                                                             """,

                                                     """
                                                                                             The value that appears most often is the Mode
                                                                                             """,

                                                     """
                                                                                             The value that appears most frequently within the data set
                                                                                             """]), ('sentence 1', ["""
                                One you have found the Median look for the value that
                                is most recurring this the Mode
                             """,

                                                                                                                    """
                                                                                                                      It is easiest to find the Mode once you have the Median.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      After you have found the Median of the data set it is
                                                                                                                      much easier to locate which value occurs the most with
                                                                                                                      the data points laid out.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      To Calculate the mode align the data points is numerical
                                                                                                                      order and find the data point the occurs the most often.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      The Mode is easiest to locate after the median is located.
                                                                                                                      Once you have the data points laid out locate the value
                                                                                                                      that recurrs the most often
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      Put the data points from least to greatest and count
                                                                                                                      the value that recurrs the most
                                                                                                                    """, ]),
                                     ('sentence 1', ["""
                                A dataset can contain more contain more than one mode.
                                If there are only recurring values the data set is Bimodal.
                             """,

                                                     """
                                                                                             If the dataset has two recurring values then it is Bimodal
                                                                                             """,

                                                     """
                                                                                             Often a data set may contain more than one recurring value.
                                                                                             In such cases the data set is referred to as bimodal
                                                                                             """,

                                                     """
                                                                                             A bimodal data set is a set that contains two Modal Values
                                                                                             """,

                                                     """
                                                                                             A data set containing two modal values is a Bimodal data set
                                                                                             """,

                                                     """
                                                                                             If two values recurr throughout the data an equal amount of
                                                                                             times the data set is Bimodal
                                                                                             """]), ('sentence 2', ["""
                                If the dataset has three recurring values then the data set is tri modal
                             """,

                                                                                                                    """
                                                                                                                      When the data set has three recurring values it is Trimodal
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If the data set contains three recurring values. The data set is trimodal
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A trimodal data set is a data set that contains three modal values
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A data set containing three modal values is a Trimodal
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If three values recur throughout the data an equal amount of times
                                                                                                                      the dataset is trimodal
                                                                                                                    """]),
                                     ('sentence 3', ["""
                              If the dataset has more than three recurring values then the data set is multi modal
                           """,

                                                     """
                                                                                              When the data set has above three recurring values. The data set is multi modal
                                                                                             """,

                                                     """
                                                                                              In larger data sets you may see multiple recurring data sets,
                                                                                              In such cases the data set is multi modal
                                                                                             """,

                                                     """
                                                                                              A multi modal set is a dataset that contains multiple recurring values
                                                                                             """,

                                                     """
                                                                                              A data set containing multiple values is Multimodal
                                                                                             """,

                                                     """
                                                                                              If multiple values recur throughout the dataset the set is multimodal
                                                                                             """])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Skew', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                    The Skew of a data set is the measure of the lack of
                                    symmetry within the data. The Skewness of the data
                                    can be either positive, negative, or undefined
                                  """,

                                                     """
                                                                                               Skewness describes the symmetry of a distribution.
                                                                                               Or lack there of symmetry. If the data has a tail end
                                                                                               distribution then the data is skewed.
                                                                                             """,

                                                     """
                                                                                               The Distribution of a data set is the Skewness of the data set.
                                                                                               This defines whether the data is symmetrical or not
                                                                                             """,

                                                     """
                                                                                               The skew defines the distribution of the data set.
                                                                                             """,

                                                     """
                                                                                               Skewness is when the curve of the data is distorted to
                                                                                               either the left or the right
                                                                                             """,

                                                     """

                                                                                             """]), ('sentence 1', ["""
                                    If the data is skewed to the right this is known as a positive
                                    distribution. In a positive distribution the mean usually greater
                                    than the median, both are to the right of the Mode.
                                  """,

                                                                                                                    """
                                                                                                                      If the tail of the data extends to the right then the data
                                                                                                                      is positively skewed. In a positively skewed distribution
                                                                                                                      the Mean is usually to the right of the median and both are
                                                                                                                      to the right of the Mode.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      When the tail of the data lays to the right of the curve.
                                                                                                                      The data has a positive skew. In a positive skew the Mean is
                                                                                                                      greater than the median and both are to the right of the mode.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If the right side of the data has a tail then the data is positively
                                                                                                                      skewed. The mean and the median lie to the right of the mode.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A curve aligned to the right extending a tail to the left is a
                                                                                                                      negatively skewed curve.
                                                                                                                    """,

                                                                                                                    """

                                                                                                                    """, ]),
                                     ('sentence 1', ["""
                                    If the data is skewed to the left this is known as a negative distribution.
                                    In a negative distribution the Mean is less than the median and both are
                                    less than the mode.
                                  """,

                                                     """
                                                                                               If the tail of the data extends to the left then the data is negatively skewed.
                                                                                               In a negative distribution the Mean is to the left of the median and both are
                                                                                               to the left of the mode.
                                                                                             """,

                                                     """
                                                                                               When the tail of the data lays to the left of the curve the mean
                                                                                               is less than the median and both are to the left of the curve.
                                                                                             """,

                                                     """
                                                                                               If the left side of the data has a tail end.
                                                                                               The data is negatively skewed.
                                                                                               The mean is less than the median and the Mode.
                                                                                             """,

                                                     """
                                                                                               A curve aligned to the left extending a tail to the
                                                                                               right is a positively skewed curve.
                                                                                             """,

                                                     """

                                                                                             """]), ('sentence 2', ["""
                                    The distribution of the data is symmetrical when the data lacks skewness.
                                  """,

                                                                                                                    """
                                                                                                                      When the data does not contain a tail it is a bell curve.
                                                                                                                      In a bell curve the data is symmetrical.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If there is no extending tail in the data the data is a Bell Curve.
                                                                                                                      A symmetrical data set
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A data set is symmetrical if it looks the same on the
                                                                                                                      left and the right side.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A curve in the center of the data is a bell curve,
                                                                                                                      a symmetrical curve.
                                                                                                                    """,

                                                                                                                    """

                                                                                                                    """]),
                                     ('sentence 3', ["""
                                The Skew is calculated as follows:
                              """, ]), ('sentence 5', [ExpressionGenerator.skew_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                ),
            }),

            ('Kurtosis', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                  Kurtosis is the measure of sharpnesss in the peak of the distribution.
                                  If it is a positive, negative, or normal curve.
                               """,

                                                     """
                                                                                                   The measure of the shape of the curve.
                                                                                                   Kurtosis measures if it is flat, normal, or peaked
                                                                                                 """,

                                                     """
                                                                                                   Kurtosis is the measure of the shape of the distribution.
                                                                                                   A measure of Kurtosis can either be a 0, a negative, or a positive number
                                                                                                 """,

                                                     """
                                                                                                   Kurtosis measures how flat or peaked out data distribution is.
                                                                                                 """,

                                                     """
                                                                                                   Kurtosis refers to the peakedness or flatness of the distribution.
                                                                                                 """,

                                                     """

                                                                                                 """]), ('sentence 1', ["""
                                   When the data does not have a peak distribution but
                                   resembles a normal bell curve this is called Mesokurtic.
                               """,

                                                                                                                        """
                                                                                                                          When the curve has the bell curve of a normal distribution
                                                                                                                          then it is a Mesokurtic Distribution
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the calculation of Kurtosis is that of a normal distribution
                                                                                                                          then the distribution is that of a Mesokurtic
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the measure of Kurtosis is a positive distribution it will
                                                                                                                          have a extended peak. This is known as Mesokurtic.
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the distribution has a sharp peak this is a positive
                                                                                                                          kurtosis known as a mesokurtic distribution
                                                                                                                        """,

                                                                                                                        """

                                                                                                                        """, ]),
                                     ('sentence 1', ["""
                                   A negative kurtosis is a Mesokurtic distribution.
                                   This will have closer to what looks like a flat distribution.
                               """,

                                                     """
                                                                                                   When the curve is flatter than a normal bell distribution
                                                                                                   this is called a Platykurtic distribution
                                                                                                 """,

                                                     """
                                                                                                   When the calculation of Kurtosis is a negative number this
                                                                                                   results in a Platykurtic distribution. A peak that is flatter
                                                                                                   than a normal distribution.
                                                                                                 """,

                                                     """
                                                                                                   When the measure of Kurtosis has a negative distribution it
                                                                                                   will have a flatter lower curve this curve is known as Platykurtic.
                                                                                                 """,

                                                     """

                                                                                                 """,

                                                     """

                                                                                                 """]), ('sentence 2', ["""
                                   A positive kurtosis is called Leptokurtic.
                                   In a Leptokurtic distribution the data has a sharp peak
                               """,

                                                                                                                        """
                                                                                                                          When the curve is taller and skinnier than a normal
                                                                                                                          distribution it has a positive Kurtosis also known
                                                                                                                          as a Letokurtic Distribution
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the calculation of Kurtosis is a positive number
                                                                                                                          this results in a Leptokurtic distribution, or a sharp peak.
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the measure of Kutosis has a regular distribution or a
                                                                                                                          distribution of 0 it will resemble the normal bell ccurve,
                                                                                                                          also known as a Leptokurtic distribution.
                                                                                                                        """,

                                                                                                                        """

                                                                                                                        """,

                                                                                                                        """

                                                                                                                        """]),
                                     ('sentence 3', ["""
                                Kurtosis is calculated as follows:
                               """, ]), ('sentence 5', [ExpressionGenerator.kurtosis_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            })]
    )
    html = Generator.render_HTML(chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir=templateDir)


def write_chapter(jobDir=None, jobName=None, templatePath='', **kwargs):
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(jobDir, jobName, 'ErrorMetrics.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Error Metrics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""
                Error metrics compare the Regression6 analysis to the data and
                are used to quantify the uncertainty or "error" in the Regression6 analysis.
                The following are all of the error metrics included in this report.
                """,

                                    """
                            Error metrics are used to quantify the uncertainty or "error" in the Regression6 analysis
                            by comparing the Regression6 analysis with respect to the data.
                            This report includes all of the following error metrics.
                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean Absolute Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'LeftColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                MAE of a data set is the average distance between
                                each data value and the mean of a data set.
                                It describes the variation and dispersion in a data set.
                                """,

                                                     """
                                                                                                            MAE of a data set measures the average distance between
                                                                                                            each data value and the mean of a data set.
                                                                                                            It describes the variation and dispersion in a data set.
                                                                                                            """]),
                                     ('sentence 1', ["""
                                    It is the average magnitude
                                    of the errors in a set of predictions. Therefore, direction is not considered.
                                    """,

                                                     """
                                                                                                            Without their direction being considered, MAE averages the magnitude
                                                                                                            of the errors in a set of predictions.
                                                                                                            """]),
                                     ('sentence 1', ["""
                                    ####
                                    It’s the average over the test sample of the absolute
                                    differences between prediction and actual observation where
                                    all individual differences have equal weight.
                                    ####
                                    """,

                                                     """
                                                                                                            ####
                                                                                                            It’s the average over the test sample of the absolute
                                                                                                            differences between prediction and actual observation where
                                                                                                            all individual differences have equal weight.
                                                                                                            """])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                            ####
                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                            """,

                                                     """
                                                                                                            ###
                                                                                                            MAE is more appropriate if being off by 00 is just twice as bad as being off by 3, then .
                                                                                                            """]),
                                     ('sentence 1', ["""
                            MAE is calculated as follows:
                            """,

                                                     """
                                                                                                            MAE is calculated using the following formula:
                                                                                                            """]),
                                     ('sentence 1', [ExpressionGenerator.mae_string()])]
                                ))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'RightColumn', 'hide': False, 'breakAfter': True, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Bias Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'LeftColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             If the absolute value is not taken (the signs of the errors are not removed),
                             the average error becomes the Mean Bias Error (MBE) and is
                             usually intended to measure average model bias.
                             MBE can convey useful information, but should be interpreted cautiously
                             because positive and negative errors will cancel out.
                             ####
                             """,

                                                     """
                                                                                                        ####
                                                                                                        If the absolute value is not taken (the signs of the errors are not removed),
                                                                                                        the average error becomes the Mean Bias Error (MBE) and is
                                                                                                        usually intended to measure average model bias.
                                                                                                        MBE can convey useful information, but should be interpreted cautiously
                                                                                                        because positive and negative errors will cancel out.
                                                                                                        ####
                                                                                                        """]),
                                     ('sentence 1', [ExpressionGenerator.mbe_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'RightColumn', 'hide': False, 'breakAfter': True, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'LeftColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The mean squared error is calculated as follows:
                             """,

                                                     """
                                                                                                           The mean squared error is calculated using the following formula:
                                                                                                           """]),
                                     ('sentence 1', [ExpressionGenerator.mse_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'RightColumn', 'hide': False, 'breakAfter': True, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Root Mean Squared Error', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'LeftColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                             ####
                             RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                             It’s the square root of the average of squared differences between prediction and actual observation.
                             ###
                             """,

                                                     """
                                                                                                           ####
                                                                                                           RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
                                                                                                           It’s the square root of the average of squared differences between prediction and actual observation.
                                                                                                           ###
                                                                                                           """]),
                                     ('sentence 1', ["""
                             RMSE does not necessarily increase with the variance of the errors.
                             RMSE is proportional to the variance of the error magnitude frequency distribution.
                             However, it is not necessarily proporional to the error variance.
                             """,

                                                     """
                                                                                                           RMSE does not necessarily increase with the variance of the errors.
                                                                                                           RMSE is proportional to the variance of the error magnitude frequency distribution.
                                                                                                           However, it is not necessarily proporional to the error variance.
                                                                                                           """]), ]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 1', ["""
                             RMSE can be appropriate when large errors should be penalized more than smaller errors.
                             The penalty of the error changes in a non-linear way.
                             """,

                                                     """
                                                                                                           ###
                                                                                                           If being off by 00 is more than twice as bad as being off by 3, RMSE can be more appropriate
                                                                                                           in some cases penalizing large errors more.
                                                                                                           If being off by 00 is just twice as bad as being off by 3, MAE is more appropriate.
                                                                                                           ####
                                                                                                           """]),
                                     ('sentence 2', ["""
                             The root mean square error is calculated as follows:
                             """,

                                                     """
                                                                                                           The root mean square error is calculated using the following formula:
                                                                                                           """]),
                                     ('sentence 3', [ExpressionGenerator.rmse_string()])]
                                ))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ['']), ]
                                )), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'RightColumn', 'hide': False, 'breakAfter': True, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [''])]
                                ))]
                            ),
                        })]
                )
            }),

        ]
    )
    templateVars = {
        'chapterTitle':    chapterTitle, 'chapterAbstract': ReportGenerator.generate_textstruct(chapterAbstract),
        'chapterSections': ReportGenerator.generate_sentences(chapterSections), 'heavyBreak': False
    }
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=outputPath)


def write_chapter(jobDir='', jobName='', templatePath='', **kwargs):
    """
    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Sections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections contain content of Figure, Caption, or Paragraph)
    """
    chapter = ReportGenerator.Chapter('Descriptive Statistics')
    mean = create_section('Mean')
    median = create_section('Median')
    mode = create_section('Mode')
    skew = create_section('Skew')
    kurtosis = create_section('Kurtosis')
    chapter.insert_sections([mean, median, mode, skew, kurtosis])
    outputPath = os.path.join(jobDir, jobName, 'DescriptiveStatistics.html')
    templateVars = {'chapter': chapter}
    print(chapter)
    """htmlOut = ReportGenerator.render_HTML(templatePath=templatePath,
                                          templateVars=templateVars,
                                          outputPath=outputPath)"""


def write_chapter(jobDir=None, jobName=None, templatePath='', **kwargs):
    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(jobDir, jobName, 'DescriptiveStatistics.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Descriptive Statistics'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph included at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', ["""

                """,

                                    """

                            """])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            ('Mean', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The Mean of your data set is also known as the average
                                your data set.
                                """,

                                                     """
                                                                                               The Average of your data set is what is called the Mean
                                                                                               or central value of the data set.
                                                                                             """,

                                                     """
                                                                                               The Mean is the average of all data points within the
                                                                                               data set and is sometime called the Arithmetic Mean.
                                                                                             """,

                                                     """
                                                                                               One way to find the Central Tendency of your data is
                                                                                               to find the Mean (average) of your data set.
                                                                                             """, """
                                The Mean is the Average of your data set
                              """, """
                                The average of all the data points within the data
                                set is called the Mean of the the data set.
                              """]), ('sentence 1', ["""
                                The average is calculated by first adding up all
                                the data points within your data set. Take the
                                sum and divide it by the total number of data points
                                within your set.
                              """,

                                                     """
                                                       You must first add up the data points within your data
                                                       set and then divide by the total number of data points.
                                                     """,

                                                     """
                                                       The mean is calculated by adding up all the data points
                                                       and dividing by the total number of data points.
                                                     """,

                                                     """
                                                       To get the Mean you must add up all the data points. take
                                                       the sum and divide by the total number of data points.
                                                     """,

                                                     """
                                                       The Mean is the summation of the data set points divided
                                                       by the number of dataset points.
                                                     """,

                                                     """
                                                       To calculate the Average (Mean), add together all the
                                                       data points within the data set, and then divide the
                                                       sum by the total number of data points within the data set.
                                                     """,

                                                     """
                                                       Add all the values together and divide them by the number
                                                       of data points within the data set.
                                                     """]), ('sentence 1', ["""
                                Often with larger data sets it may be easier to define
                                the data set by a single number, known as the Central Tendency.
                                Calculating the Mean is one way to find this number.
                              """,

                                                                            """
                                                                              When dealing with a larger data set it may be easier to use a singular
                                                                              value to define it, this value is called the Central Tendency. Calculating
                                                                              the mean is one of the ways to find the Central Tendency.
                                                                            """,

                                                                            """
                                                                              This gives you a central number to define your data
                                                                            """, ])]
                                )), ('paragraph 1', collections.OrderedDict(
                                    [('sentence 0', ["""
                             The following formula is used for calculating Mean.
                            """,

                                                     """
                                                                                              """]), ('sentence 1', ["""
                            Mean is calculated as follows:
                            """,

                                                                                                                     """
                                                                                                                     Mean is calculated using the following formula:
                                                                                                                     """]),
                                     ('sentence 1', [ExpressionGenerator.mean_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Median', {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                 The Median is the midpoint of the data set. It is the
                                 point that separates the lower half of your data from the top half.
                              """,

                                                     """
                                                                                                 The midpoint of the data set is known as the Median. The Median is
                                                                                                 used to represent the center of your data set.
                                                                                               """,

                                                     """
                                                                                                 Your data can be split into two sets, a lower half and a top half.
                                                                                                 The midpoint of the set is known as the Median
                                                                                               """,

                                                     """
                                                                                                 The Median is the data point lying at the midpoint of the data set.
                                                                                               """,

                                                     """
                                                                                                 The Median is the center value within a data set.
                                                                                               """,

                                                     """
                                                                                                 The median is the middle value of the data set
                                                                                               """,

                                                     """
                                                                                                 The median is the middle most value of data set.
                                                                                               """]), ('sentence 1', ["""
                                If you want to find the Median of your data set,
                                set all the data points within the data set in numerical order.
                             """,

                                                                                                                      """
                                                                                                                         In order to get the Median set the data points within the data set
                                                                                                                         in numerical order.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        You can get the Median of a data set by setting out all the
                                                                                                                        data points in order.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        The Median is found by locating the midpoint within the data set.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        Order the values within the data set in numerical order.
                                                                                                                        Look for the center value.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        Lay out the data in numerical order.
                                                                                                                      """,

                                                                                                                      """
                                                                                                                        Lay out the data set from least to greatest and locate
                                                                                                                        the middle value.
                                                                                                                      """

                                                                                                                      ]),
                                     ('sentence 1', ["""
                                If your data set does not have a midpoint due to there being an
                                even number set of data points. Find the mean of the two midpoints.
                             """,

                                                     """
                                                                                                 When the data set has an even number of data points the Mean of the
                                                                                                 two most center points will be taken and that will be known as the
                                                                                                 median of the data set.
                                                                                               """,

                                                     """
                                                                                                 If the midpoint of your data set cannot be located due to the
                                                                                                 data having an odd number of data points. Then the two midpoints of
                                                                                                 the data set will be taken and the Mean of the two points will be
                                                                                                 found and become the median.
                                                                                               """,

                                                     """
                                                                                                 If the data set does not have a center point. Find the two most midpoints
                                                                                                 within the data set and find their mean. That Mean is the Median of the
                                                                                                 data set.
                                                                                               """,

                                                     """
                                                                                                 If the data has an even set of numbers and no center point then calculate
                                                                                                 the mean of the two center points.
                                                                                               """,

                                                     """
                                                                                                 If the data set has an odd number of data points the midpoint is the median.
                                                                                                 If the number of data points within the data set equates to an even number
                                                                                                 of data points. Then the Mean of the two mid points will be the median of
                                                                                                 the data set.
                                                                                               """]),
                                     ('sentence 2', [ExpressionGenerator.median_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Mode', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                The Mode of the dataset is the most reccuring data
                                point through out the data set
                              """,

                                                     """
                                                                                               The Value or frequency to appear within the data set
                                                                                               most often is referred to as the Mode
                                                                                             """,

                                                     """
                                                                                               The Mode is the value or data point that appears most
                                                                                               often within the data set
                                                                                             """,

                                                     """
                                                                                               When a data set contains a value that occurs more than
                                                                                               once it is known as the Mode. This is the Value most likely
                                                                                               to be picked fro the data set
                                                                                             """,

                                                     """
                                                                                             The value that appears most often is the Mode
                                                                                             """,

                                                     """
                                                                                             The value that appears most frequently within the data set
                                                                                             """]), ('sentence 1', ["""
                                One you have found the Median look for the value that
                                is most recurring this the Mode
                             """,

                                                                                                                    """
                                                                                                                      It is easiest to find the Mode once you have the Median.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      After you have found the Median of the data set it is
                                                                                                                      much easier to locate which value occurs the most with
                                                                                                                      the data points laid out.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      To Calculate the mode align the data points is numerical
                                                                                                                      order and find the data point the occurs the most often.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      The Mode is easiest to locate after the median is located.
                                                                                                                      Once you have the data points laid out locate the value
                                                                                                                      that recurrs the most often
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      Put the data points from least to greatest and count
                                                                                                                      the value that recurrs the most
                                                                                                                    """, ]),
                                     ('sentence 1', ["""
                                A dataset can contain more contain more than one mode.
                                If there are only recurring values the data set is Bimodal.
                             """,

                                                     """
                                                                                             If the dataset has two recurring values then it is Bimodal
                                                                                             """,

                                                     """
                                                                                             Often a data set may contain more than one recurring value.
                                                                                             In such cases the data set is referred to as bimodal
                                                                                             """,

                                                     """
                                                                                             A bimodal data set is a set that contains two Modal Values
                                                                                             """,

                                                     """
                                                                                             A data set containing two modal values is a Bimodal data set
                                                                                             """,

                                                     """
                                                                                             If two values recurr throughout the data an equal amount of
                                                                                             times the data set is Bimodal
                                                                                             """]), ('sentence 2', ["""
                                If the dataset has three recurring values then the data set is tri modal
                             """,

                                                                                                                    """
                                                                                                                      When the data set has three recurring values it is Trimodal
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If the data set contains three recurring values. The data set is trimodal
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A trimodal data set is a data set that contains three modal values
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A data set containing three modal values is a Trimodal
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If three values recur throughout the data an equal amount of times
                                                                                                                      the dataset is trimodal
                                                                                                                    """]),
                                     ('sentence 3', ["""
                              If the dataset has more than three recurring values then the data set is multi modal
                           """,

                                                     """
                                                                                              When the data set has above three recurring values. The data set is multi modal
                                                                                             """,

                                                     """
                                                                                              In larger data sets you may see multiple recurring data sets,
                                                                                              In such cases the data set is multi modal
                                                                                             """,

                                                     """
                                                                                              A multi modal set is a dataset that contains multiple recurring values
                                                                                             """,

                                                     """
                                                                                              A data set containing multiple values is Multimodal
                                                                                             """,

                                                     """
                                                                                              If multiple values recur throughout the dataset the set is multimodal
                                                                                             """])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            }),

            ('Skew', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                    The Skew of a data set is the measure of the lack of
                                    symmetry within the data. The Skewness of the data
                                    can be either positive, negative, or undefined
                                  """,

                                                     """
                                                                                               Skewness describes the symmetry of a distribution.
                                                                                               Or lack there of symmetry. If the data has a tail end
                                                                                               distribution then the data is skewed.
                                                                                             """,

                                                     """
                                                                                               The Distribution of a data set is the Skewness of the data set.
                                                                                               This defines whether the data is symmetrical or not
                                                                                             """,

                                                     """
                                                                                               The skew defines the distribution of the data set.
                                                                                             """,

                                                     """
                                                                                               Skewness is when the curve of the data is distorted to
                                                                                               either the left or the right
                                                                                             """,

                                                     """

                                                                                             """]), ('sentence 1', ["""
                                    If the data is skewed to the right this is known as a positive
                                    distribution. In a positive distribution the mean usually greater
                                    than the median, both are to the right of the Mode.
                                  """,

                                                                                                                    """
                                                                                                                      If the tail of the data extends to the right then the data
                                                                                                                      is positively skewed. In a positively skewed distribution
                                                                                                                      the Mean is usually to the right of the median and both are
                                                                                                                      to the right of the Mode.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      When the tail of the data lays to the right of the curve.
                                                                                                                      The data has a positive skew. In a positive skew the Mean is
                                                                                                                      greater than the median and both are to the right of the mode.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If the right side of the data has a tail then the data is positively
                                                                                                                      skewed. The mean and the median lie to the right of the mode.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A curve aligned to the right extending a tail to the left is a
                                                                                                                      negatively skewed curve.
                                                                                                                    """,

                                                                                                                    """

                                                                                                                    """, ]),
                                     ('sentence 1', ["""
                                    If the data is skewed to the left this is known as a negative distribution.
                                    In a negative distribution the Mean is less than the median and both are
                                    less than the mode.
                                  """,

                                                     """
                                                                                               If the tail of the data extends to the left then the data is negatively skewed.
                                                                                               In a negative distribution the Mean is to the left of the median and both are
                                                                                               to the left of the mode.
                                                                                             """,

                                                     """
                                                                                               When the tail of the data lays to the left of the curve the mean
                                                                                               is less than the median and both are to the left of the curve.
                                                                                             """,

                                                     """
                                                                                               If the left side of the data has a tail end.
                                                                                               The data is negatively skewed.
                                                                                               The mean is less than the median and the Mode.
                                                                                             """,

                                                     """
                                                                                               A curve aligned to the left extending a tail to the
                                                                                               right is a positively skewed curve.
                                                                                             """,

                                                     """

                                                                                             """]), ('sentence 2', ["""
                                    The distribution of the data is symmetrical when the data lacks skewness.
                                  """,

                                                                                                                    """
                                                                                                                      When the data does not contain a tail it is a bell curve.
                                                                                                                      In a bell curve the data is symmetrical.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      If there is no extending tail in the data the data is a Bell Curve.
                                                                                                                      A symmetrical data set
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A data set is symmetrical if it looks the same on the
                                                                                                                      left and the right side.
                                                                                                                    """,

                                                                                                                    """
                                                                                                                      A curve in the center of the data is a bell curve,
                                                                                                                      a symmetrical curve.
                                                                                                                    """,

                                                                                                                    """

                                                                                                                    """]),
                                     ('sentence 3', ["""
                                The Skew is calculated as follows:
                              """, ]), ('sentence 5', [ExpressionGenerator.skew_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', [''])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                ),
            }),

            ('Kurtosis', {
                'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict(
                        [('sentence 0', ['']),

                         ]
                    )), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        ('Definition', {
                            'column':         'CenterColumn', 'hide': False, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', ["""
                                  Kurtosis is the measure of sharpnesss in the peak of the distribution.
                                  If it is a positive, negative, or normal curve.
                               """,

                                                     """
                                                                                                   The measure of the shape of the curve.
                                                                                                   Kurtosis measures if it is flat, normal, or peaked
                                                                                                 """,

                                                     """
                                                                                                   Kurtosis is the measure of the shape of the distribution.
                                                                                                   A measure of Kurtosis can either be a 0, a negative, or a positive number
                                                                                                 """,

                                                     """
                                                                                                   Kurtosis measures how flat or peaked out data distribution is.
                                                                                                 """,

                                                     """
                                                                                                   Kurtosis refers to the peakedness or flatness of the distribution.
                                                                                                 """,

                                                     """

                                                                                                 """]), ('sentence 1', ["""
                                   When the data does not have a peak distribution but
                                   resembles a normal bell curve this is called Mesokurtic.
                               """,

                                                                                                                        """
                                                                                                                          When the curve has the bell curve of a normal distribution
                                                                                                                          then it is a Mesokurtic Distribution
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the calculation of Kurtosis is that of a normal distribution
                                                                                                                          then the distribution is that of a Mesokurtic
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the measure of Kurtosis is a positive distribution it will
                                                                                                                          have a extended peak. This is known as Mesokurtic.
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the distribution has a sharp peak this is a positive
                                                                                                                          kurtosis known as a mesokurtic distribution
                                                                                                                        """,

                                                                                                                        """

                                                                                                                        """, ]),
                                     ('sentence 1', ["""
                                   A negative kurtosis is a Mesokurtic distribution.
                                   This will have closer to what looks like a flat distribution.
                               """,

                                                     """
                                                                                                   When the curve is flatter than a normal bell distribution
                                                                                                   this is called a Platykurtic distribution
                                                                                                 """,

                                                     """
                                                                                                   When the calculation of Kurtosis is a negative number this
                                                                                                   results in a Platykurtic distribution. A peak that is flatter
                                                                                                   than a normal distribution.
                                                                                                 """,

                                                     """
                                                                                                   When the measure of Kurtosis has a negative distribution it
                                                                                                   will have a flatter lower curve this curve is known as Platykurtic.
                                                                                                 """,

                                                     """

                                                                                                 """,

                                                     """

                                                                                                 """]), ('sentence 2', ["""
                                   A positive kurtosis is called Leptokurtic.
                                   In a Leptokurtic distribution the data has a sharp peak
                               """,

                                                                                                                        """
                                                                                                                          When the curve is taller and skinnier than a normal
                                                                                                                          distribution it has a positive Kurtosis also known
                                                                                                                          as a Letokurtic Distribution
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the calculation of Kurtosis is a positive number
                                                                                                                          this results in a Leptokurtic distribution, or a sharp peak.
                                                                                                                        """,

                                                                                                                        """
                                                                                                                          When the measure of Kutosis has a regular distribution or a
                                                                                                                          distribution of 0 it will resemble the normal bell ccurve,
                                                                                                                          also known as a Leptokurtic distribution.
                                                                                                                        """,

                                                                                                                        """

                                                                                                                        """,

                                                                                                                        """

                                                                                                                        """]),
                                     ('sentence 3', ["""
                                Kurtosis is calculated as follows:
                               """, ]), ('sentence 5', [ExpressionGenerator.kurtosis_string()])]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])])), ]
                            ),
                        }),  # SUBSECTION ---------------
                        ('Example', {
                            'column':         'CenterColumn', 'hide': True, 'breakAfter': False, 'displayTitle': True,
                            'preFigures':     collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure 1', collections.OrderedDict(
                                    [('image', ''), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', [''])]))]
                            ),
                        })]
                )
            })]
    )
    templateVars = {
        'chapterTitle':    chapterTitle, 'chapterAbstract': ReportGenerator.generate_textstruct(chapterAbstract),
        'chapterSections': ReportGenerator.generate_sentences(chapterSections), 'heavyBreak': False
    }
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=outputPath)


def write_chapter(jobDir=None, jobName=None, templatePath='', **kwargs):
    """
    report_generator0 Structure:
        CHAPTER (Chapters can contain any number of Sections)
            SECTION (Sections can contain any number of Subsections)
                SUBSECTION (Subsections contain content of Figure, Caption, or Paragraph)
    """

    outputPath = os.path.join(jobDir, jobName, 'DescriptiveStatistics.html')

    chapterTitle = 'Descriptive Statistics'

    templateVars = {
        'chapterTitle':    chapterTitle, 'chapterAbstract': ReportGenerator.generate_textstruct(chapterAbstract),
        'chapterSections': ReportGenerator.generate_sentences(chapterSections), 'heavyBreak': False
    }
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=outputPath)


def write_chapter(data, dtypes=None, path=None, pageNumber=8, templateDir=None):
    """
    :param vars: list of strings
    :param dtypes: list of strings
    :return:
    """
    imagesPath = os.path.join(os.path.join(path, 'HTML2'), 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = generate_all_plots(data, dtypes, imagesPath, N=1)
    plots = collections.OrderedDict(
        [(var, ['Scatter plot', 'Distribution plot', 'Distribution Fit plot', 'Box plot',
                'Violin plot'] if dtype == 'Numerical' else ['Bar plot']) for var, dtype in zip(vars, dtypes)]
    )

    end = '. '
    captions = {}
    postText = {}
    columns = {
        'Bar Plot':         'CenterColumn', 'Scatter Plot': 'CenterColumn', 'Box Plot': 'LeftColumn',
        'Violin Plot':      'RightColumn', 'Distribution Plot': 'LeftColumn', 'Distribution Fit Plot': 'RightColumn',
        'Correlation Plot': 'CenterColumn'
    }

    def bar():
        captions['Bar Plot'] = {}
        postText['Bar Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Categorical':
                captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
                postText['Bar Plot'][var] = random.choice(
                    ["""
                                                    This is a bar plot0.""", """
                                                           Welcome to the bar plot0.
                                                           """]
                )

    def box():
        captions['Box Plot'] = {}
        postText['Box Plot'] = {}

        for var, dtype in zip(vars, dtypes):
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

                postText['Box Plot'][var] = boxText

    def distribution():
        captions['Distribution Plot'] = {}
        postText['Distribution Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                distributionText = ''
                captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
                distributionText += random.choice(
                    ['This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'denotes the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'indicates the regions where {} ' \
                     'is concentrated. '.format(var),

                     'This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'indicates the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'denotes the regions where {} ' \
                     'is concentrated. '.format(var), ]
                )
                distributionText += random.choice(['', ''])
                postText['Distribution Plot'][var] = distributionText

    def distribution_fit():
        captions['Distribution Fit Plot'] = {}
        postText['Distribution Fit Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                keys = list(distFit[var].keys())
                func1 = distFit[var][keys[0]][0]
                # func2 = distFit[var][keys[1]][0]
                error1 = s_round(distFit[var][keys[0]][1])
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

                postText['Distribution Fit Plot'][var] = distributionFitText

    def scatter():
        captions['Scatter Plot'] = {}
        postText['Scatter Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                fit = s_round(regression[var][0])
                errors = regression[var][1]
                mse = s_round(errors['mse'])
                R2 = s_round(errors['R2'])
                rmse = s_round(errors['rmse'])
                mae = s_round(errors['mae'])
                sor = s_round(errors['sum of residuals'])
                chi2 = s_round(errors['chi2'])
                reducedChi2 = s_round(errors['reduced chi2'])
                standardError = s_round(errors['standard error'])
                captions['Scatter Plot'][var] = 'Scatter plot0 and Regression6 fit of <var>{0}</var>.'.format(var)
                scatterText = ''
                scatterText += random.choice(
                    ['Using curve fitting procedures, '
                     '<var>{0}</var> was found to be best described by $${1}$$'.format(
                        var, fit
                    ),

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
                scatterText += random.choice(
                    ['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)]
                )
                scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
                scatterText += random.choice(
                    ['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)]
                )
                scatterText += random.choice(
                    ['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)]
                )
                scatterText += random.choice(
                    [
                        'The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
                        'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
                )
                scatterText += random.choice(
                    ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
                     'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(
                        standardError, sor
                    )]
                )

                postText['Scatter Plot'][var] = scatterText

    def violin():
        captions['Violin Plot'] = {}
        postText['Violin Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
                # CHECK STATISTICS
                mean = s_round(numStats[var]['Mean'])
                median = s_round(numStats[var]['Median'])
                mode = s_round(numStats[var]['Mode'])
                std = s_round(numStats[var]['Standard\nDeviation'])
                variance = s_round(numStats[var]['Variance'])
                skew = s_round(numStats[var]['Skew'])
                kurtosis = s_round(numStats[var]['Kurtosis'])
                x = s_round(mean - median)
                c0 = s_round(mean / median)
                c1 = s_round((abs(x) / mean) * 100)

                # MEAN ----------------------------------------------
                meanSentence = random.choice(
                    ['The distribution of <var>' + var + '</var> has a <em>mean</em> of ' + str(mean),

                     '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
                ) + end

                # MEDIAN --------------------------------------------
                medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
                if x > 0:
                    """Mean is greater than median"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % greater than ' \
                                           'the <em>median</em>. '.format(str(c0), str(c1))
                else:
                    """Median is greater than mean"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % smaller than ' \
                                           'the <em>median</em>. '.format(str(s_round(1 / c0)), str(c1))

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
                    """fairly symmetrical"""
                    skewText += random.choice(['slightly'])
                elif abs(skew) <= 1:
                    """moderately skew"""
                    skewText += random.choice(['moderately'])
                else:
                    """highly skewed"""
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

                postText['Violin Plot'][var] = violinText

    def correlation():
        captions['Correlation Plot'] = {}
        postText['Correlation Plot'] = {}
        if 'Numerical' in dtypes:
            captions['Correlation Plot'] = 'Correlation heatmap. '
            correlationText = ''
            correlationText += random.choice(['This is the correlation matrix of the data plotted as a heatmap. ', ])
            postText['Correlation Plot'] = correlationText

    bar()
    box()
    distribution()
    distribution_fit()
    scatter()
    violin()
    correlation()

    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(path, 'HTML2'), 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[:-1], {
                            'column':       columns[plot[:-1]], 'pageNumber': (
                                pageNumber + (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]) - sum(
                                [len([0 for p in plots[v] if columns[p[:-1]] == 'LeftColumn']) for k, v in
                                 enumerate(vars) if k < j]
                            ) - sum(
                                [1 for m, plot in enumerate(plots[var]) if m < i and columns[plot[:-1]] == 'LeftColumn']
                            ) - 1), 'hide': False, 'breakAfter': True, 'displayTitle': False,
                            'preText':      collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ), 'figures':   collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(
                                            Paths.PLOTS_DIR, plot.replace(
                                                ' ', ''
                                            )[:-5]
                                        ), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':  collections.OrderedDict(
                                [('paragraph 0',
                                  collections.OrderedDict([('sentence 0', [postText[plot[:-1]][var]]), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    if 'Numerical' in dtypes:
        chapterSections.update(
            {
                'Correlation': {
                    'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                    'abstract':     collections.OrderedDict(
                        [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                    ),  # SUBSECTIONS --------------
                    'subSections':  collections.OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'column':      'CenterColumn', 'pageNumber': pageNumber + sum(
                                    [len(plots[v]) for v in vars]
                                ) - sum(
                                    [len(
                                        [0 for p in plots[v] if columns[p[:-1]] == 'LeftColumn']
                                    ) for v in vars]
                                ), 'hide':     False, 'breakAfter': True, 'displayTitle': False,
                                'preText':     collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', ['']), ]
                                    )), ]
                                ), 'figures':  collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            os.path.join(
                                                Paths.PLOTS_DIR, 'Heatmap'
                                            ), 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                    1 + sum(
                                                        [len(
                                                            plots[v]
                                                        ) for v in vars]
                                                    ), captions['Correlation Plot']
                                                )])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'postText': collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', [postText['Correlation Plot']]), ]
                                    )), ]
                                ),
                            })]
                    )
                }
            }
        )
        chapterSections.move_to_end('Correlation', last=True)
    chapterAbstract = Generator.generate_textStruct(chapterAbstract)
    chapterSections = Generator.generate_sentences(chapterSections)
    htmlOut = Generator.render_HTML(
        chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir=templateDir, heavyBreak=False
    )


def write_chapter(data=None, dtypes=None, path=None, pageNumber=8, templateDir=None):
    """
    :param vars: list of strings
    :param dtypes: list of strings
    :return:
    """
    imagesPath = os.path.join(os.path.join(path, 'HTML2'), 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = generate_all_plots(data, dtypes, imagesPath, N=1)
    plots = collections.OrderedDict(
        [(var, [('Scatter plot', 'Regression6 Error Tables'), ('Distribution plot', 'Central Tendencies Tables'),
                ('Distribution Fit plot', 'Distribution Error Tables'), ('Box plot', 'IQR Tables'),
                ('Violin plot', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar plot', '')]) for var, dtype in
         zip(vars, dtypes)]
    )

    end = '. '
    captions = {}
    preText = {}
    postText = {}
    columns = {
        'Bar Plot':         'CenterColumn', 'Scatter Plot': 'CenterColumn', 'Box Plot': 'LeftColumn',
        'Violin Plot':      'RightColumn', 'Distribution Plot': 'LeftColumn', 'Distribution Fit Plot': 'RightColumn',
        'Correlation Plot': 'CenterColumn'
    }

    def bar():
        captions['Bar Plot'] = {}
        preText['Bar Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Categorical':
                captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
                preText['Bar Plot'][var] = random.choice(
                    ["""
                                                    This is a bar plot0.""", """
                                                          Welcome to the bar plot0.
                                                          """]
                )

    def box():
        captions['Box Plot'] = {}
        preText['Box Plot'] = {}

        for var, dtype in zip(vars, dtypes):
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

    def distribution():
        captions['Distribution Plot'] = {}
        preText['Distribution Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                distributionText = ''
                captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
                distributionText += random.choice(
                    ['This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'denotes the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'indicates the regions where {} ' \
                     'is concentrated. '.format(var),

                     'This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'indicates the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'denotes the regions where {} ' \
                     'is concentrated. '.format(var), ]
                )
                distributionText += random.choice(['', ''])
                preText['Distribution Plot'][var] = distributionText

    def distribution_fit():
        captions['Distribution Fit Plot'] = {}
        preText['Distribution Fit Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                keys = list(distFit[var].keys())
                func1 = distFit[var][keys[0]][0]
                # func2 = distFit[var][keys[1]][0]
                error1 = s_round(distFit[var][keys[0]][1])
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

    def scatter():
        captions['Scatter Plot'] = {}
        preText['Scatter Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                fit = s_round(regression[var][0])
                errors = regression[var][1]
                mse = s_round(errors['MSE'])
                R2 = s_round(errors['R Squared'])
                rmse = s_round(errors['RMSE'])
                mae = s_round(errors['MAE'])
                sor = s_round(errors['Sum of Residuals'])
                chi2 = s_round(errors['Chi Squared'])
                reducedChi2 = s_round(errors['Reduced Chi Squared'])
                standardError = s_round(errors['Standard Error'])
                captions['Scatter Plot'][var] = 'Scatter plot0 and Regression6 fit of <var>{0}</var>.'.format(var)
                scatterText = ''
                scatterText += random.choice(
                    ['Using curve fitting procedures, '
                     '<var>{0}</var> was found to be best described by $${1}$$'.format(
                        var, fit
                    ),

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
                scatterText += random.choice(
                    ['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)]
                )
                scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
                scatterText += random.choice(
                    ['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)]
                )
                scatterText += random.choice(
                    ['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)]
                )
                scatterText += random.choice(
                    [
                        'The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
                        'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
                )
                scatterText += random.choice(
                    ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
                     'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(
                        standardError, sor
                    )]
                )

                preText['Scatter Plot'][var] = scatterText

    def violin():
        captions['Violin Plot'] = {}
        preText['Violin Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
                # CHECK STATISTICS
                mean = s_round(numStats[var]['Mean'])
                median = s_round(numStats[var]['Median'])
                mode = s_round(numStats[var]['Mode'])
                std = s_round(numStats[var]['Standard Deviation'])
                variance = s_round(numStats[var]['Variance'])
                skew = s_round(numStats[var]['Skew'])
                kurtosis = s_round(numStats[var]['Kurtosis'])
                x = s_round(mean - median)
                c0 = s_round(mean / median)
                c1 = s_round((abs(x) / mean) * 100)

                # MEAN ----------------------------------------------
                meanSentence = random.choice(
                    ['The distribution of <var>' + var + '</var> has a <em>mean</em> of ' + str(mean),

                     '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
                ) + end

                # MEDIAN --------------------------------------------
                medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
                if x > 0:
                    """Mean is greater than median"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % greater than ' \
                                           'the <em>median</em>. '.format(str(c0), str(c1))
                else:
                    """Median is greater than mean"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % smaller than ' \
                                           'the <em>median</em>. '.format(str(s_round(1 / c0)), str(c1))

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
                    """fairly symmetrical"""
                    skewText += random.choice(['slightly'])
                elif abs(skew) <= 1:
                    """moderately skew"""
                    skewText += random.choice(['moderately'])
                else:
                    """highly skewed"""
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

    def correlation():
        captions['Correlation Plot'] = {}
        preText['Correlation Plot'] = {}
        if 'Numerical' in dtypes:
            captions['Correlation Plot'] = 'Correlation heatmap. '
            correlationText = ''
            correlationText += random.choice(['This is the correlation matrix of the data plotted as a heatmap. ', ])
            preText['Correlation Plot'] = correlationText

    bar()
    box()
    distribution()
    distribution_fit()
    scatter()
    violin()
    correlation()

    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(path, 'HTML2'), 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'column':         columns[plot[0][:-1]], 'pageNumber': (
                                pageNumber + (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]) - sum(
                                [len([0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']) for k, v in
                                 enumerate(vars) if k < j]
                            ) - sum(
                                [1 for m, plot in enumerate(plots[var]) if
                                 m < i and columns[plot[0][:-1]] == 'LeftColumn']
                            ) - 1), 'hide':   False, 'breakAfter': True, 'displayTitle': False,
                            'preFigures':     collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(Paths.PLOTS_DIR, plot[0].replace(' ', '')[:-5]), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [preText[plot[0][:-1]][var]]), ]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(
                                            Paths.PLOTS_DIR, plot[1].replace(' ', '')[:-6]
                                        ), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Table {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(plots[v]) for k, v in enumerate(vars) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    if 'Numerical' in dtypes:
        chapterSections.update(
            {
                'Correlation': {
                    'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                    'abstract':     collections.OrderedDict(
                        [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                    ),  # SUBSECTIONS --------------
                    'subSections':  collections.OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'column':         'CenterColumn', 'pageNumber': pageNumber + sum(
                                    [len(plots[v]) for v in vars]
                                ) - sum(
                                    [len(
                                        [0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']
                                    ) for v in vars]
                                ), 'hide':        False, 'breakAfter': True, 'displayTitle': False,
                                'preFigures':     collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            os.path.join(
                                                Paths.PLOTS_DIR, 'Heatmap'
                                            ), 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                    1 + sum(
                                                        [len(
                                                            plots[v]
                                                        ) for v in vars]
                                                    ), captions['Correlation Plot']
                                                )])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'preText':     collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', [preText['Correlation Plot']]), ]
                                    )), ]
                                ), 'postFigures': collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            os.path.join(
                                                Paths.PLOTS_DIR, 'Heatmap'
                                            ), 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', [''])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'postText':    collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', ['']), ]
                                    )), ]
                                ),
                            })]
                    )
                }
            }
        )
        chapterSections.move_to_end('Correlation', last=True)
    htmlOut = Generator.render_HTML(
        chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir=templateDir, heavyBreak=False
    )


def write_chapter(data=None, dtypes=None, path=None, jobName=None, pageNumber=8, templateDir=None):
    """
    :param vars: list of strings
    :param dtypes: list of strings
    :return:
    """
    imagesPath = os.path.join(os.path.join(path, jobName), 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = generate_all_plots(data, dtypes, imagesPath, N=1)
    plots = collections.OrderedDict(
        [(var, [('Scatter plot', 'Regression6 Error Tables'), ('Distribution plot', 'Central Tendencies Tables'),
                ('Distribution Fit plot', 'Distribution Error Tables'), ('Box plot', 'IQR Tables'),
                ('Violin plot', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar plot', '')]) for var, dtype in
         zip(vars, dtypes)]
    )

    end = '. '
    captions = {}
    preText = {}
    postText = {}
    columns = {
        'Bar Plot':         'CenterColumn', 'Scatter Plot': 'CenterColumn', 'Box Plot': 'LeftColumn',
        'Violin Plot':      'RightColumn', 'Distribution Plot': 'LeftColumn', 'Distribution Fit Plot': 'RightColumn',
        'Correlation Plot': 'CenterColumn'
    }

    def bar():
        captions['Bar Plot'] = {}
        preText['Bar Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Categorical':
                captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
                preText['Bar Plot'][var] = random.choice(
                    ["""
                                                    This is a bar plot0.""", """
                                                          Welcome to the bar plot0.
                                                          """]
                )

    def box():
        captions['Box Plot'] = {}
        preText['Box Plot'] = {}

        for var, dtype in zip(vars, dtypes):
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

    def distribution():
        captions['Distribution Plot'] = {}
        preText['Distribution Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                distributionText = ''
                captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
                distributionText += random.choice(
                    ['This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'denotes the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'indicates the regions where {} ' \
                     'is concentrated. '.format(var),

                     'This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'indicates the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'denotes the regions where {} ' \
                     'is concentrated. '.format(var), ]
                )
                distributionText += random.choice(['', ''])
                preText['Distribution Plot'][var] = distributionText

    def distribution_fit():
        captions['Distribution Fit Plot'] = {}
        preText['Distribution Fit Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                keys = list(distFit[var].keys())
                func1 = distFit[var][keys[0]][0]
                # func2 = distFit[var][keys[1]][0]
                error1 = s_round(distFit[var][keys[0]][1])
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

    def scatter():
        captions['Scatter Plot'] = {}
        preText['Scatter Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                fit = s_round(regression[var][0])
                errors = regression[var][1]
                mse = s_round(errors['MSE'])
                R2 = s_round(errors['R Squared'])
                rmse = s_round(errors['RMSE'])
                mae = s_round(errors['MAE'])
                sor = s_round(errors['Sum of Residuals'])
                chi2 = s_round(errors['Chi Squared'])
                reducedChi2 = s_round(errors['Reduced Chi Squared'])
                standardError = s_round(errors['Standard Error'])
                captions['Scatter Plot'][var] = 'Scatter plot0 and Regression6 fit of <var>{0}</var>.'.format(var)
                scatterText = ''
                scatterText += random.choice(
                    ['Using curve fitting procedures, '
                     '<var>{0}</var> was found to be best described by $${1}$$'.format(
                        var, fit
                    ),

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
                scatterText += random.choice(
                    ['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)]
                )
                scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
                scatterText += random.choice(
                    ['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)]
                )
                scatterText += random.choice(
                    ['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)]
                )
                scatterText += random.choice(
                    [
                        'The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
                        'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
                )
                scatterText += random.choice(
                    ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
                     'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(
                        standardError, sor
                    )]
                )

                preText['Scatter Plot'][var] = scatterText

    def violin():
        captions['Violin Plot'] = {}
        preText['Violin Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
                # CHECK STATISTICS
                mean = s_round(numStats[var]['Mean'])
                median = s_round(numStats[var]['Median'])
                mode = s_round(numStats[var]['Mode'])
                std = s_round(numStats[var]['Standard Deviation'])
                variance = s_round(numStats[var]['Variance'])
                skew = s_round(numStats[var]['Skew'])
                kurtosis = s_round(numStats[var]['Kurtosis'])
                x = s_round(mean - median)
                c0 = s_round(mean / median)
                c1 = s_round((abs(x) / mean) * 100)

                # MEAN ----------------------------------------------
                meanSentence = random.choice(
                    ['The distribution of <var>' + var + '</var> has a <em>mean</em> of ' + str(mean),

                     '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
                ) + end

                # MEDIAN --------------------------------------------
                medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
                if x > 0:
                    """Mean is greater than median"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % greater than ' \
                                           'the <em>median</em>. '.format(str(c0), str(c1))
                else:
                    """Median is greater than mean"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % smaller than ' \
                                           'the <em>median</em>. '.format(str(s_round(1 / c0)), str(c1))

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
                    """fairly symmetrical"""
                    skewText += random.choice(['slightly'])
                elif abs(skew) <= 1:
                    """moderately skew"""
                    skewText += random.choice(['moderately'])
                else:
                    """highly skewed"""
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

    def correlation():
        captions['Correlation Plot'] = {}
        preText['Correlation Plot'] = {}
        if 'Numerical' in dtypes:
            captions['Correlation Plot'] = 'Correlation heatmap. '
            correlationText = ''
            correlationText += random.choice(['This is the correlation matrix of the data plotted as a heatmap. ', ])
            preText['Correlation Plot'] = correlationText

    bar()
    box()
    distribution()
    distribution_fit()
    scatter()
    violin()
    correlation()

    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(path, jobName), 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'column':         columns[plot[0][:-1]], 'pageNumber': (
                                pageNumber + (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]) - sum(
                                [len([0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']) for k, v in
                                 enumerate(vars) if k < j]
                            ) - sum(
                                [1 for m, plot in enumerate(plots[var]) if
                                 m < i and columns[plot[0][:-1]] == 'LeftColumn']
                            ) - 1), 'hide':   False, 'breakAfter': True, 'displayTitle': False,
                            'preFigures':     collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(Paths.PLOTS_DIR, plot[0].replace(' ', '')[:-5]), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [preText[plot[0][:-1]][var]]), ]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(
                                            Paths.PLOTS_DIR, plot[1].replace(' ', '')[:-6]
                                        ), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Table {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(plots[v]) for k, v in enumerate(vars) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    if 'Numerical' in dtypes:
        chapterSections.update(
            {
                'Correlation': {
                    'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                    'abstract':     collections.OrderedDict(
                        [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                    ),  # SUBSECTIONS --------------
                    'subSections':  collections.OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'column':         'CenterColumn', 'pageNumber': pageNumber + sum(
                                    [len(plots[v]) for v in vars]
                                ) - sum(
                                    [len(
                                        [0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']
                                    ) for v in vars]
                                ), 'hide':        False, 'breakAfter': True, 'displayTitle': False,
                                'preFigures':     collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            os.path.join(
                                                Paths.PLOTS_DIR, 'Heatmap'
                                            ), 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                    1 + sum(
                                                        [len(
                                                            plots[v]
                                                        ) for v in vars]
                                                    ), captions['Correlation Plot']
                                                )])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'preText':     collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', [preText['Correlation Plot']]), ]
                                    )), ]
                                ), 'postFigures': collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            os.path.join(
                                                Paths.PLOTS_DIR, 'Heatmap'
                                            ), 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', [''])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'postText':    collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', ['']), ]
                                    )), ]
                                ),
                            })]
                    )
                }
            }
        )
        chapterSections.move_to_end('Correlation', last=True)
    htmlOut = Generator.render_HTML(
        chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir=templateDir, heavyBreak=False
    )


def write_chapter(data=None, dtypes=None, path=None, jobName=None, pageNumber=8, templateDir=None):
    """
    vars: list of strings
    dtypes: list of strings
    return:
    """
    imagesPath = os.path.join(os.path.join(path, jobName), 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = StaticPlotGenerator.generate_all_plots(
        data, dtypes, imagesPath, N=1
    )
    plots = collections.OrderedDict(
        [(var, [('Scatter plot', 'Regression6 Error Tables'), ('Distribution plot', 'Central Tendencies Tables'),
                ('Distribution Fit plot', 'Distribution Error Tables'), ('Box plot', 'IQR Tables'),
                ('Violin plot', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar plot', '')]) for var, dtype in
         zip(vars, dtypes)]
    )

    end = '. '
    captions = {}
    preText = {}
    postText = {}
    columns = {
        'Bar Plot':         'CenterColumn', 'Scatter Plot': 'CenterColumn', 'Box Plot': 'LeftColumn',
        'Violin Plot':      'RightColumn', 'Distribution Plot': 'LeftColumn', 'Distribution Fit Plot': 'RightColumn',
        'Correlation Plot': 'CenterColumn'
    }

    def bar():
        captions['Bar Plot'] = {}
        preText['Bar Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Categorical':
                captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
                preText['Bar Plot'][var] = random.choice(
                    ["""
                                                    This is a bar plot0.""", """
                                                          Welcome to the bar plot0.
                                                          """]
                )

    def box():
        captions['Box Plot'] = {}
        preText['Box Plot'] = {}

        for var, dtype in zip(vars, dtypes):
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

    def distribution():
        captions['Distribution Plot'] = {}
        preText['Distribution Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                distributionText = ''
                captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
                distributionText += random.choice(
                    ['This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'denotes the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'indicates the regions where {} ' \
                     'is concentrated. '.format(var),

                     'This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'indicates the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'denotes the regions where {} ' \
                     'is concentrated. '.format(var), ]
                )
                distributionText += random.choice(['', ''])
                preText['Distribution Plot'][var] = distributionText

    def distribution_fit():
        captions['Distribution Fit Plot'] = {}
        preText['Distribution Fit Plot'] = {}
        for var, dtype in zip(vars, dtypes):
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

    def scatter():
        captions['Scatter Plot'] = {}
        preText['Scatter Plot'] = {}
        for var, dtype in zip(vars, dtypes):
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
                     '<var>{0}</var> was found to be best described by $${1}$$'.format(
                        var, fit
                    ),

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
                scatterText += random.choice(
                    ['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)]
                )
                scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
                scatterText += random.choice(
                    ['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)]
                )
                scatterText += random.choice(
                    ['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)]
                )
                scatterText += random.choice(
                    [
                        'The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
                        'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
                )
                scatterText += random.choice(
                    ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
                     'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(
                        standardError, sor
                    )]
                )

                preText['Scatter Plot'][var] = scatterText

    def violin():
        captions['Violin Plot'] = {}
        preText['Violin Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
                # CHECK STATISTICS
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

                     '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
                ) + end

                # MEDIAN --------------------------------------------
                medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
                if x > 0:
                    """Mean is greater than median"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % greater than ' \
                                           'the <em>median</em>. '.format(str(c0), str(c1))
                else:
                    """Median is greater than mean"""
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
                    """fairly symmetrical"""
                    skewText += random.choice(['slightly'])
                elif abs(skew) <= 1:
                    """moderately skew"""
                    skewText += random.choice(['moderately'])
                else:
                    """highly skewed"""
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

    def correlation():
        captions['Correlation Plot'] = {}
        preText['Correlation Plot'] = {}
        if 'Numerical' in dtypes:
            captions['Correlation Plot'] = 'Correlation heatmap. '
            correlationText = ''
            correlationText += random.choice(['This is the correlation matrix of the data plotted as a heatmap. ', ])
            preText['Correlation Plot'] = correlationText

    bar()
    box()
    distribution()
    distribution_fit()
    scatter()
    violin()
    correlation()

    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(os.path.join(path, jobName), 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [

            # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'column':         columns[plot[0][:-1]], 'pageNumber': (
                                pageNumber + (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]) - sum(
                                [len([0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']) for k, v in
                                 enumerate(vars) if k < j]
                            ) - sum(
                                [1 for m, plot in enumerate(plots[var]) if
                                 m < i and columns[plot[0][:-1]] == 'LeftColumn']
                            ) - 1), 'hide':   False, 'breakAfter': True, 'displayTitle': False,
                            'preFigures':     collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(ReportPaths.PLOTS_DIR, plot[0].replace(' ', '')[:-5]), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(plots[v]) for k, v in enumerate(vars) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0',
                                  collections.OrderedDict([('sentence 0', [preText[plot[0][:-1]][var]]), ])), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        os.path.join(
                                            ReportPaths.PLOTS_DIR, plot[1].replace(' ', '')[:-6]
                                        ), var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Table {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(plots[v]) for k, v in enumerate(vars) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    if 'Numerical' in dtypes:
        chapterSections.update(
            {
                'Correlation': {
                    'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                    'abstract':     collections.OrderedDict(
                        [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                    ),  # SUBSECTIONS --------------
                    'subSections':  collections.OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'column':         'CenterColumn', 'pageNumber': pageNumber + sum(
                                    [len(plots[v]) for v in vars]
                                ) - sum(
                                    [len(
                                        [0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']
                                    ) for v in vars]
                                ), 'hide':        False, 'breakAfter': True, 'displayTitle': False,
                                'preFigures':     collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            os.path.join(
                                                ReportPaths.PLOTS_DIR, 'Heatmap'
                                            ), 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                    1 + sum(
                                                        [len(
                                                            plots[v]
                                                        ) for v in vars]
                                                    ), captions['Correlation Plot']
                                                )])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'preText':     collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', [preText['Correlation Plot']]), ]
                                    )), ]
                                ), 'postFigures': collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', ''), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', [''])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'postText':    collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', ['']), ]
                                    )), ]
                                ),
                            })]
                    )
                }
            }
        )
        chapterSections.move_to_end('Correlation', last=True)
    htmlOut = ReportGenerator.render_HTML(
        chapterTitle, chapterAbstract, chapterSections, outputPath, templateDir=templateDir, heavyBreak=False
    )


def write_chapter(data=None, dtypes=None, jobDir=None, jobName=None, pageNumber=8, templatePath='', **kwargs):
    """
    vars: list of strings
    dtypes: list of strings
    return:
    """
    imagesPath = os.path.join(jobDir, jobName, 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = StaticPlotGenerator.generate_all_plots(
        data, dtypes, imagesPath, N=1
    )
    plots = collections.OrderedDict(
        [(var, [('Scatter plot', 'Regression6 Error Tables'), ('Distribution plot', 'Central Tendencies Tables'),
                ('Distribution Fit plot', 'Distribution Error Tables'), ('Box plot', 'IQR Tables'),
                ('Violin plot', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar plot', '')]) for var, dtype in
         zip(vars, dtypes)]
    )
    for var, dtype in zip(vars, dtypes):
        bar(var, dtype)
        box(var, dtype)
        distribution(var, dtype)
        distribution_fit(var, dtype)
        scatter(var, dtype)
        violin(var, dtype)
        correlation()
    outputPath = os.path.join(jobDir, jobName, 'Analysis.html')

    chapterSections = collections.OrderedDict(
        [  # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'column':         columns[plot[0][:-1]], 'pageNumber': (
                                pageNumber + (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]) - sum(
                                [len([0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']) for k, v in
                                 enumerate(vars) if k < j]
                            ) - sum(
                                [1 for m, plot in enumerate(plots[var]) if
                                 m < i and columns[plot[0][:-1]] == 'LeftColumn']
                            ) - 1), 'hide':   False, 'breakAfter': True, 'displayTitle': False,
                            'preFigures':     collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        ReportPaths.PLOTS_DIR, plot[0].replace(' ', '')[:-5], var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [preText[plot[0][:-1]][var]]), ]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        ReportPaths.PLOTS_DIR, plot[1].replace(' ', '')[:-6], var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Table {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    if 'Numerical' in dtypes:
        chapterSections.update(
            {
                'Correlation': {
                    'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                    'abstract':     collections.OrderedDict(
                        [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                    ),  # SUBSECTIONS --------------
                    'subSections':  collections.OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'column':         'CenterColumn', 'pageNumber': pageNumber + sum(
                                    [len(plots[v]) for v in vars]
                                ) - sum(
                                    [len(
                                        [0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']
                                    ) for v in vars]
                                ), 'hide':        False, 'breakAfter': False, 'displayTitle': False,
                                'preFigures':     collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            ReportPaths.PLOTS_DIR, 'Heatmap', 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                    1 + sum(
                                                        [len(
                                                            plots[v]
                                                        ) for v in vars]
                                                    ), captions['Correlation Plot']
                                                )])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'preText':     collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', [preText['Correlation Plot']]), ]
                                    )), ]
                                ), 'postFigures': collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', ''), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', [''])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'postText':    collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', ['']), ]
                                    )), ]
                                ),
                            })]
                    )
                }
            }
        )
    chapterSections.move_to_end('Correlation', last=True)
    templateVars = {'chapter': chapter}
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=outputPath)


def write_chapter(data=None, dtypes=None, jobDir=None, jobName=None, pageNumber=8, templatePath='', **kwargs):
    """
    vars: list of strings
    dtypes: list of strings
    return:
    """
    imagesPath = os.path.join(jobDir, jobName, 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = StaticPlotGenerator.generate_all_plots(
        data, dtypes, imagesPath, N=1
    )
    plots = collections.OrderedDict(
        [(var, [('Scatter plot', 'Regression6 Error Tables'), ('Distribution plot', 'Central Tendencies Tables'),
                ('Distribution Fit plot', 'Distribution Error Tables'), ('Box plot', 'IQR Tables'),
                ('Violin plot', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar plot', '')]) for var, dtype in
         zip(vars, dtypes)]
    )

    end = '. '
    captions = {}
    preText = {}
    postText = {}
    columns = {
        'Bar Plot':         'CenterColumn', 'Scatter Plot': 'CenterColumn', 'Box Plot': 'LeftColumn',
        'Violin Plot':      'RightColumn', 'Distribution Plot': 'LeftColumn', 'Distribution Fit Plot': 'RightColumn',
        'Correlation Plot': 'CenterColumn'
    }

    def bar():
        captions['Bar Plot'] = {}
        preText['Bar Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Categorical':
                captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
                preText['Bar Plot'][var] = random.choice(
                    ["""
                                                    This is a bar plot0.""", """
                                                          Welcome to the bar plot0.
                                                          """]
                )

    def box():
        captions['Box Plot'] = {}
        preText['Box Plot'] = {}

        for var, dtype in zip(vars, dtypes):
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

    def distribution():
        captions['Distribution Plot'] = {}
        preText['Distribution Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                distributionText = ''
                captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
                distributionText += random.choice(
                    ['This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'denotes the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'indicates the regions where {} ' \
                     'is concentrated. '.format(var),

                     'This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'indicates the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'denotes the regions where {} ' \
                     'is concentrated. '.format(var), ]
                )
                distributionText += random.choice(['', ''])
                preText['Distribution Plot'][var] = distributionText

    def distribution_fit():
        captions['Distribution Fit Plot'] = {}
        preText['Distribution Fit Plot'] = {}
        for var, dtype in zip(vars, dtypes):
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

    def scatter():
        captions['Scatter Plot'] = {}
        preText['Scatter Plot'] = {}
        for var, dtype in zip(vars, dtypes):
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
                     '<var>{0}</var> was found to be best described by $${1}$$'.format(
                        var, fit
                    ),

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
                scatterText += random.choice(
                    ['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)]
                )
                scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
                scatterText += random.choice(
                    ['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)]
                )
                scatterText += random.choice(
                    ['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)]
                )
                scatterText += random.choice(
                    [
                        'The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
                        'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
                )
                scatterText += random.choice(
                    ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
                     'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(
                        standardError, sor
                    )]
                )

                preText['Scatter Plot'][var] = scatterText

    def violin():
        captions['Violin Plot'] = {}
        preText['Violin Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
                # CHECK STATISTICS
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

                     '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
                ) + end

                # MEDIAN --------------------------------------------
                medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
                if x > 0:
                    """Mean is greater than median"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % greater than ' \
                                           'the <em>median</em>. '.format(str(c0), str(c1))
                else:
                    """Median is greater than mean"""
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
                    """fairly symmetrical"""
                    skewText += random.choice(['slightly'])
                elif abs(skew) <= 1:
                    """moderately skew"""
                    skewText += random.choice(['moderately'])
                else:
                    """highly skewed"""
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

    def correlation():
        captions['Correlation Plot'] = {}
        preText['Correlation Plot'] = {}
        if 'Numerical' in dtypes:
            captions['Correlation Plot'] = 'Correlation heatmap. '
            correlationText = ''
            correlationText += random.choice(['This is the correlation matrix of the data plotted as a heatmap. ', ])
            preText['Correlation Plot'] = correlationText

    bar()
    box()
    distribution()
    distribution_fit()
    scatter()
    violin()
    correlation()

    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(jobDir, jobName, 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [  # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'column':         columns[plot[0][:-1]], 'pageNumber': (
                                pageNumber + (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]) - sum(
                                [len([0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']) for k, v in
                                 enumerate(vars) if k < j]
                            ) - sum(
                                [1 for m, plot in enumerate(plots[var]) if
                                 m < i and columns[plot[0][:-1]] == 'LeftColumn']
                            ) - 1), 'hide':   False, 'breakAfter': True, 'displayTitle': False,
                            'preFigures':     collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        ReportPaths.PLOTS_DIR, plot[0].replace(' ', '')[:-5], var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [preText[plot[0][:-1]][var]]), ]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        ReportPaths.PLOTS_DIR, plot[1].replace(' ', '')[:-6], var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Table {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    if 'Numerical' in dtypes:
        chapterSections.update(
            {
                'Correlation': {
                    'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                    'abstract':     collections.OrderedDict(
                        [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                    ),  # SUBSECTIONS --------------
                    'subSections':  collections.OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'column':         'CenterColumn', 'pageNumber': pageNumber + sum(
                                    [len(plots[v]) for v in vars]
                                ) - sum(
                                    [len(
                                        [0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']
                                    ) for v in vars]
                                ), 'hide':        False, 'breakAfter': False, 'displayTitle': False,
                                'preFigures':     collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            ReportPaths.PLOTS_DIR, 'Heatmap', 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                    1 + sum(
                                                        [len(
                                                            plots[v]
                                                        ) for v in vars]
                                                    ), captions['Correlation Plot']
                                                )])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'preText':     collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', [preText['Correlation Plot']]), ]
                                    )), ]
                                ), 'postFigures': collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', ''), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', [''])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'postText':    collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', ['']), ]
                                    )), ]
                                ),
                            })]
                    )
                }
            }
        )
    chapterSections.move_to_end('Correlation', last=True)
    templateVars = {
        'chapterTitle':    chapterTitle, 'chapterAbstract': ReportGenerator.generate_textstruct(chapterAbstract),
        'chapterSections': ReportGenerator.generate_sentences(chapterSections), 'heavyBreak': False
    }
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=outputPath)

    # from LoParDatasets.SyntheticData.SyntheticData import data as data
    """import pandas as pd

    data = pd.read_csv(r'Z:\Family\Transfer\sensor_data.csv')
    data.set_index(data.columns[0], inplace=True)
    data = DataProcessor.optimize_dataFrame(data)
    dtypes = DataProcessor.get_dtypes(data)
    write_chapter(data, dtypes)"""


def write_chapter(data=None, dtypes=None, jobDir=None, jobName=None, pageNumber=8, templatePath='', **kwargs):
    """
    vars: list of strings
    dtypes: list of strings
    return:
    """
    imagesPath = os.path.join(jobDir, jobName, 'StaticPlots0')
    vars, numStats, catStats, distFit, regression = StaticPlotGenerator.generate_all_plots(
        data, dtypes, imagesPath, N=1
    )
    plots = collections.OrderedDict(
        [(var, [('Scatter plot', 'Regression6 Error Tables'), ('Distribution plot', 'Central Tendencies Tables'),
                ('Distribution Fit plot', 'Distribution Error Tables'), ('Box plot', 'IQR Tables'),
                ('Violin plot', 'Dispersion Tables')] if dtype == 'Numerical' else [('Bar plot', '')]) for var, dtype in
         zip(vars, dtypes)]
    )

    end = '. '
    captions = {}
    preText = {}
    postText = {}
    columns = {
        'Bar Plot':         'CenterColumn', 'Scatter Plot': 'CenterColumn', 'Box Plot': 'LeftColumn',
        'Violin Plot':      'RightColumn', 'Distribution Plot': 'LeftColumn', 'Distribution Fit Plot': 'RightColumn',
        'Correlation Plot': 'CenterColumn'
    }

    def bar():
        captions['Bar Plot'] = {}
        preText['Bar Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Categorical':
                captions['Bar Plot'][var] = 'Bar Plot of <var>{0}</var>.'.format(var)
                preText['Bar Plot'][var] = random.choice(
                    ["""
                                                    This is a bar plot0.""", """
                                                          Welcome to the bar plot0.
                                                          """]
                )

    def box():
        captions['Box Plot'] = {}
        preText['Box Plot'] = {}

        for var, dtype in zip(vars, dtypes):
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

    def distribution():
        captions['Distribution Plot'] = {}
        preText['Distribution Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                distributionText = ''
                captions['Distribution Plot'][var] = 'Histogram of <var>{0}</var>.'.format(var)
                distributionText += random.choice(
                    ['This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'denotes the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'indicates the regions where {} ' \
                     'is concentrated. '.format(var),

                     'This plot0 shows the frequency '
                     'distribution of the data. '
                     'The rug at the bottom of the plot0 '
                     'indicates the concentration of the data. ',

                     'This plot0 shows the frequency ' \
                     'distribution of the data. ' \
                     'The rug at the bottom of the plot0 ' \
                     'denotes the regions where {} ' \
                     'is concentrated. '.format(var), ]
                )
                distributionText += random.choice(['', ''])
                preText['Distribution Plot'][var] = distributionText

    def distribution_fit():
        captions['Distribution Fit Plot'] = {}
        preText['Distribution Fit Plot'] = {}
        for var, dtype in zip(vars, dtypes):
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

    def scatter():
        captions['Scatter Plot'] = {}
        preText['Scatter Plot'] = {}
        for var, dtype in zip(vars, dtypes):
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
                     '<var>{0}</var> was found to be best described by $${1}$$'.format(
                        var, fit
                    ),

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
                scatterText += random.choice(
                    ['The <em>mean squared error</em> of this fit is <em>{}</em>. '.format(mse)]
                )
                scatterText += random.choice(['The <em>R squared</em> of this fit is <em>{}</em>. '.format(R2)])
                scatterText += random.choice(
                    ['The <em>root mean squared error</em> of this fit is <em>{}</em>. '.format(rmse)]
                )
                scatterText += random.choice(
                    ['The <em>mean absolute error</em> of this fit is <em>{}</em>. '.format(mae)]
                )
                scatterText += random.choice(
                    [
                        'The Regression6 also has a <em>Chi Squared</em> value of <em>{0}</em> and a <em>Reduced Chi Squared</em> '
                        'value of <em>{1}</em>. '.format(chi2, reducedChi2)]
                )
                scatterText += random.choice(
                    ['Further, the <em>Standard Error</em> of the Regression6 is <em>{0}</em> '
                     'and the <em>Sum of Residuals</em> is <em>{1}</em>. '.format(
                        standardError, sor
                    )]
                )

                preText['Scatter Plot'][var] = scatterText

    def violin():
        captions['Violin Plot'] = {}
        preText['Violin Plot'] = {}
        for var, dtype in zip(vars, dtypes):
            if dtype == 'Numerical':
                captions['Violin Plot'][var] = 'Violin plot0 of <var>{0}</var>.'.format(var)
                # CHECK STATISTICS
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

                     '<var>' + var + '</var> has a distribution ' + 'with a mean of <em>' + str(mean) + '</em>']
                ) + end

                # MEDIAN --------------------------------------------
                medianSentence = random.choice(['The <em>median</em> is {}. '.format(str(median))])
                if x > 0:
                    """Mean is greater than median"""
                    meanMedianComparison = 'The <em>mean</em> is {0} ' \
                                           'times, or {1} % greater than ' \
                                           'the <em>median</em>. '.format(str(c0), str(c1))
                else:
                    """Median is greater than mean"""
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
                    """fairly symmetrical"""
                    skewText += random.choice(['slightly'])
                elif abs(skew) <= 1:
                    """moderately skew"""
                    skewText += random.choice(['moderately'])
                else:
                    """highly skewed"""
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

    def correlation():
        captions['Correlation Plot'] = {}
        preText['Correlation Plot'] = {}
        if 'Numerical' in dtypes:
            captions['Correlation Plot'] = 'Correlation heatmap. '
            correlationText = ''
            correlationText += random.choice(['This is the correlation matrix of the data plotted as a heatmap. ', ])
            preText['Correlation Plot'] = correlationText

    bar()
    box()
    distribution()
    distribution_fit()
    scatter()
    violin()
    correlation()

    """
    REPORT STRUCTURE

    CHAPTER (Chapters can contain any number of Sections)
    |Chapter Title
    |Chapter Abstract
    |
    |SECTION (Sections can contain any number of Subsections)
    ||
    ||Section Title
    ||Section Abstract
    ||
    ||SUBSECTION
    |||
    |||Subsection Title
    |||Subsection Pretext
    |||Subsection Image
    |||Subsection Caption
    |||Subsection PostText
    """
    # OUTPUT PATH ----------------------
    # The output path is the path where
    # the HTML2 report will be configuration.py.
    # outputPath = os.path.join('HTML2', %%file name goes here%%)
    outputPath = os.path.join(jobDir, jobName, 'Analysis.html')

    # CHAPTER TITLE --------------------
    # The title of the chapter is
    # a mandatory string.
    # Chapter Title HTML2: <h2>
    chapterTitle = 'Analysis'

    # CHAPTER ABSTRACT -----------------
    # The chapter abstract is a
    # paragraph include at the beginning
    # of the chapter.
    # It's a dictionary of dictionaries of lists of strings.
    # The key in the dictionary is a paragraph (dictionary).
    # Each paragraph is a dictionary.
    # The key in the paragraph (dictionary) is a sentence (list).
    # Each sentence is represented by a list of strings.
    # Each string in a list represents
    # a variation of that sentence.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}
    # Chapter Abstract HTML2: <p object_oriented="ChapterAbstract">
    # OUTER DICTIONARY -----------------
    chapterAbstract = collections.OrderedDict(
        [  # PARAGRAPH: DICTIONARY OF SENTENCES
            ('paragraph 0', collections.OrderedDict(
                [  # SENTENCE: LIST OF STRINGS
                    ('sentence 0', [''])]
            ))]
    )

    # CHAPTER SECTIONS -----------------
    # chapterSections is a dictionary.
    # Each key in chapterSections is a section:
    # 'section 1' = {'displayTitle': Boolean,
    #                    'abstract': {'sentence 0': ['']}
    #                    'subSections': {}}

    # 'displayTitle' determines
    # whether the title is displayed or not.

    # 'abstract' contains a lists of strings.
    # The abstract structure is identical
    # to the chapterAbstract.
    # The abstract is a paragraph to be displayed
    # at the beginning of the section.

    # 'subSections' contains a dictionary of subSections.
    # 'subSections' = {'sub section A': {}}

    # SUB SECTIONS %%%%%%%%%%%%%%%%%%%%%
    # A subSection is a dictionary:
    # 'sub section A' = {'displayTitle': Boolean,
    #                   'preText': {},
    #                   'image': '',
    #                   'caption': '',
    #                   'postText': {}}

    # 'displayTitle' determines whether
    # the subSection title is displayed.

    # 'preText' contains text to be displayed
    # at the beginning of the subSection,
    # before the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%
    # A figure is a dictionary.
    # 'figures': {'figure 0':
    #               {'image': path,
    #               'caption': string}}
    # 'image' is a string.
    # The string is the path to an image to be
    # displayed between the preText and postText.
    # This is optional, and will be ignored if
    # set to an empty string ''.
    #
    # 'caption' is a string.
    # The string is a caption to accompany the
    # corresponding image.
    # This is optional, and will be ignored if
    # set to an empty string ''.

    # 'postText' contains text to be
    # displayed at the end of a subSection,
    # after the image/caption.
    # Its structure is identical to chapterAbstract.
    # This is optional, and will be ignored if
    # set to {'': {'': ['']}}

    # Section title HTML2: <h3>
    # Section abstract HTML2: <p object_oriented="SectionAbstract">
    # Subsection title HTML2: <h4>
    # Subsection text HTML2: <p object_oriented="SubSectionText">
    # Subsection image HTML2: <image object_oriented="SubSectionImage">
    # Subsection caption HTML2: <p object_oriented="SubSectionImageCaption">
    chapterSections = collections.OrderedDict(
        [  # SECTION ----------------------
            (var, {
                'displayTitle': True, 'breakAfter': True,  # SECTION ABSTRACT ---------
                'abstract':     collections.OrderedDict(
                    [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                ),  # SUBSECTIONS --------------
                'subSections':  collections.OrderedDict(
                    [  # SUBSECTION -----------
                        (plot[0][:-1], {
                            'column':         columns[plot[0][:-1]], 'pageNumber': (
                                pageNumber + (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j]) - sum(
                                [len([0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']) for k, v in
                                 enumerate(vars) if k < j]
                            ) - sum(
                                [1 for m, plot in enumerate(plots[var]) if
                                 m < i and columns[plot[0][:-1]] == 'LeftColumn']
                            ) - 1), 'hide':   False, 'breakAfter': True, 'displayTitle': False,
                            'preFigures':     collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        ReportPaths.PLOTS_DIR, plot[0].replace(' ', '')[:-5], var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'preText':     collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict(
                                    [('sentence 0', [preText[plot[0][:-1]][var]]), ]
                                )), ]
                            ), 'postFigures': collections.OrderedDict(
                                [('Figure {}'.format(
                                    (i + 1) + sum([len(plots[v]) for k, v in enumerate(vars) if k < j])
                                ), collections.OrderedDict(
                                    [('image', os.path.join(
                                        ReportPaths.PLOTS_DIR, plot[1].replace(' ', '')[:-6], var + '.png'
                                    )), ('caption', collections.OrderedDict(
                                        [('paragraph 0', collections.OrderedDict(
                                            [('sentence 0', ['<var>Table {0}:</var> {1}'.format(
                                                (i + 1) + sum(
                                                    [len(
                                                        plots[v]
                                                    ) for k, v in enumerate(
                                                        vars
                                                    ) if k < j]
                                                ), captions[plot[0][:-1]][var]
                                            )])]
                                        ))]
                                    ))]
                                )), ]
                            ), 'postText':    collections.OrderedDict(
                                [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                            ),
                        }) for i, plot in enumerate(plots[var])]
                )
            }) for j, var in enumerate(vars)]
    )

    if 'Numerical' in dtypes:
        chapterSections.update(
            {
                'Correlation': {
                    'displayTitle': True, 'breakAfter': False,  # SECTION ABSTRACT ---------
                    'abstract':     collections.OrderedDict(
                        [('paragraph 0', collections.OrderedDict([('sentence 0', ['']), ])), ]
                    ),  # SUBSECTIONS --------------
                    'subSections':  collections.OrderedDict(
                        [  # SUBSECTION -----------
                            ('Correlation Plot', {
                                'column':         'CenterColumn', 'pageNumber': pageNumber + sum(
                                    [len(plots[v]) for v in vars]
                                ) - sum(
                                    [len(
                                        [0 for p in plots[v] if columns[p[0][:-1]] == 'LeftColumn']
                                    ) for v in vars]
                                ), 'hide':        False, 'breakAfter': False, 'displayTitle': False,
                                'preFigures':     collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', os.path.join(
                                            ReportPaths.PLOTS_DIR, 'Heatmap', 'Correlation' + '.png'
                                        )), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', ['<var>Figure {0}:</var> {1}'.format(
                                                    1 + sum(
                                                        [len(
                                                            plots[v]
                                                        ) for v in vars]
                                                    ), captions['Correlation Plot']
                                                )])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'preText':     collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', [preText['Correlation Plot']]), ]
                                    )), ]
                                ), 'postFigures': collections.OrderedDict(
                                    [('Figure -1', collections.OrderedDict(
                                        [('image', ''), ('caption', collections.OrderedDict(
                                            [('paragraph 0', collections.OrderedDict(
                                                [('sentence 0', [''])]
                                            ))]
                                        ))]
                                    )), ]
                                ), 'postText':    collections.OrderedDict(
                                    [('paragraph 0', collections.OrderedDict(
                                        [('sentence 0', ['']), ]
                                    )), ]
                                ),
                            })]
                    )
                }
            }
        )
    chapterSections.move_to_end('Correlation', last=True)
    templateVars = {
        'chapterTitle':    chapterTitle, 'chapterAbstract': ReportGenerator.generate_textstruct(chapterAbstract),
        'chapterSections': ReportGenerator.generate_sentences(chapterSections), 'heavyBreak': False
    }
    htmlOut = ReportGenerator.render_HTML(templatePath=templatePath, templateVars=templateVars, outputPath=outputPath)

    # from LoParDatasets.SyntheticData.SyntheticData import data as data
    """import pandas as pd

    data = pd.read_csv(r'Z:\Family\Transfer\sensor_data.csv')
    data.set_index(data.columns[0], inplace=True)
    data = DataProcessor.optimize_dataFrame(data)
    dtypes = DataProcessor.get_dtypes(data)
    write_chapter(data, dtypes)"""


openFile()
"""object_oriented _Chapter(OrderedDict):
    def __repr__(self):
        r = 'Chapter {}\n'.format(self['title'])
        for key in self:
            if key != 'sections':
                r += '   {0}: {1}\n'.format(key, self[key])
        if self['sections'].__len__() > 0:
            for section in self['sections']:
                r += '\t{}'.format(self['sections'][section])
        return r

    def insert_sections(self, sections):
        newSections = []
        for section in self['sections']:
            newSections.append((section, self['sections'][section]))
        for section in sections:
            newSections.append((section['title'], section))
        self['sections'] = OrderedDict(newSections)


def Chapter(title, sections=[]):
    _c = [
        ('title', title),
        ('sections',
         OrderedDict([(section['title'], section) for section in sections]))
    ]
    c = _Chapter(_c)
    return c
    """
"""object_oriented _Chapter(OrderedDict):
    def __repr__(self):
        r = 'Chapter {}\n'.format(self['title'])
        for key in self:
            if key != 'sections':
                r += '   {0}: {1}\n'.format(key, self[key])
        if self['sections'].__len__() > 0:
            for section in self['sections']:
                r += '\t{}'.format(self['sections'][section])
        return r

    def insert_sections(self, sections):
        newSections = []
        for section in self['sections']:
            newSections.append((section, self['sections'][section]))
        for section in sections:
            newSections.append((section['title'], section))
        self['sections'] = OrderedDict(newSections)


def Chapter(title, sections=[]):
    _c = [
        ('title', title),
        ('sections',
         OrderedDict([(section['title'], section) for section in sections]))
    ]
    c = _Chapter(_c)
    return c"""
"""
def randstr():
    s = ''
    for m in range(random.randint(2, 3)):
        for j in range(random.randint(6, 12)):
            if j == 0:
                s += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            k = random.randint(2, 3)
            for i in range(k):
                s += random.choice('abcdefghijklmnopqrstuvwxyz')
            s += ' '
        s += '. '
    return s


def collect_chapters(dtypes):
    """
    dtypes: list of str
    return:
    """
    chapters = [plot]
    if 'Categorical' in dtypes:
        pass
    if 'Numerical' in dtypes:
        chapters.append(ErrorMetrics)
        chapters.append(Quartiles_IQR)
        chapters.append(DescriptiveStatistics)
    paragraphs = {chapter: [] for chapter in chapters}
    return chapters, paragraphs
"""
"""
def randstr():
    s = ''
    for m in range(random.randint(2, 3)):
        for j in range(random.randint(6, 12)):
            if j == 0:
                s += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            k = random.randint(2, 3)
            for i in range(k):
                s += random.choice('abcdefghijklmnopqrstuvwxyz')
            s += ' '
        s += '. '
    return s


def collect_chapters(dtypes):
    """
    dtypes: list of str
    return:
    """
    chapters = [plot]
    if 'Categorical' in dtypes:
        pass
    if 'Numerical' in dtypes:
        chapters.append(ErrorMetrics)
        chapters.append(Quartiles_IQR)
        chapters.append(DescriptiveStatistics)
    paragraphs = {chapter: [] for chapter in chapters}
    return chapters, paragraphs
"""
'''
