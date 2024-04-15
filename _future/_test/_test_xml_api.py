from _misc.lozoya.xml import *  # change to import lozoya.xml_api

# soup = get_soup(r'http://www.cnn.com/2017/11/29/politics/president-donald-trump-competency/index.html')
# paragraphs = get_paragraphs(soup)
# formatter('WAR', paragraphs)
# x = get_soup('https://www.wikipedia.org/wiki/Philosophy')
# px = get_paragraphs(x)
# formatter('death', px, width=75)
print("Starting...")
root = 'Z:\Family\LoPar Technologies\LoParJobs\StatisticalAnalysisReportJobs'
jobID = 'bb49aa85-ac79-41ef-8097-07194f0e277a'
parser = TextParser()
soup_original_1 = bs("".join(open(r"{}\{}\Test report_generator0\Analysis.html".format(root, jobID))))
soup_original_2 = bs(''.join(open(r"{}\{}\Test report_generator0\DescriptiveStatistics.html".fornat(root, jobID))))
for element in soup_original_2:
    soup_original_1.append(copy.deepcopy(element))
f = open(r'{}\{}\Test report_generator0\Analysis.html'.format(root, jobID), 'w')
m = str(soup_original_1)
f.write(m)
f.close()
