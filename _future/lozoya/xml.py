import os
import sys
import urllib.request
from html.parser import HTMLParser
from urllib import parse
from urllib.request import urlopen

from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
from bs4 import BeautifulSoup as bs
from z__legacy import TextParser

print("Starting...")
parser = TextParser()

import urllib.request


def get_soup(url):
    sauce = urllib.request.urlopen(url).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    return soup


def get_paragraphs(soup):
    paragraphs = []
    for paragraph in soup.find_all('p'):
        paragraphs.append(paragraph.text)
    return paragraphs


def get_links(soup):
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))
    return links


def get_tables(soup):
    tables = []
    for table in soup.find_all('table'):
        tables.append(table)
    return tables


def get_all(soup):
    content = {}
    paragraphs = get_paragraphs(soup)
    links = get_links(soup)
    tables = get_tables(soup)
    content['paragraphs'] = paragraphs
    content['links'] = links
    content['tables'] = tables
    return content


def formatter(token, paragraphs, width=100):
    for paragraph in paragraphs:
        if token.lower() in paragraph.lower():
            line = 1
            placeholder = 0
            for e, char in enumerate(paragraph):
                if e != len(paragraph) - 1:
                    if (e > line * width and paragraph[e + 1] == ' '):
                        line += 1
                        print(paragraph[placeholder:e + 1])
                        placeholder = e + 2
                else:
                    print(paragraph[placeholder:])


import urllib.request

import bs4 as bs


def get_soup(url):
    sauce = urllib.request.urlopen(url).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    return soup


def get_paragraphs(soup):
    paragraphs = []
    for paragraph in soup.find_all('p'):
        paragraphs.append(paragraph.text)
    return paragraphs


def get_links(soup):
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))
    return links


def get_tables(soup):
    tables = []
    for table in soup.find_all('table'):
        tables.append(table)
    return tables


def get_all(soup):
    content = {}
    paragraphs = get_paragraphs(soup)
    links = get_links(soup)
    tables = get_tables(soup)
    content['paragraphs'] = paragraphs
    content['links'] = links
    content['tables'] = tables
    return content


def formatter(token, paragraphs, width=100):
    for paragraph in paragraphs:
        if token.lower() in paragraph.lower():
            line = 1
            placeholder = 0
            for e, char in enumerate(paragraph):
                if e != len(paragraph) - 1:
                    if (e > line * width and paragraph[e + 1] == ' '):
                        line += 1
                        print(paragraph[placeholder:e + 1])
                        placeholder = e + 2
                else:
                    print(paragraph[placeholder:])


# We are going to create a object_oriented called LinkParser that inherits some
# methods from HTMLParser which is why it is passed into the definition
class LinkParser(HTMLParser):

    # This is a function that HTMLParser normally has
    # but we are adding some functionality to it
    def handle_starttag(self, tag, attrs):
        # We are looking for the begining of a link. Links normally look
        # like <a href="www.someurl.com"></a>
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    # We are grabbing the new URL. We are also adding the
                    # base URL to it. For test:
                    # www.netinstructions.com is the base and
                    # somepage.html is the new URL (a relative URL)
                    #
                    # We combine a relative URL with the base URL to create
                    # an absolute URL like:
                    # www.netinstructions.com/somepage.html
                    newUrl = parse.urljoin(self.baseUrl, value)
                    # And add it to our colection of links:
                    self.links = self.links + [newUrl]

    # This is a new function that we are creating to get links
    # that our spider() function will call
    def getLinks(self, url):
        self.links = []
        # Remember the base URL which will be important when creating
        # absolute URLs
        self.baseUrl = url
        # Use the urlopen function from the standard Python 2 library
        response = urlopen(url)
        # Make sure that we are looking at HTML2 and not search things that
        # are floating around on the internet (such as
        # JavaScript files, CSS, or .PDFs for test)
        if response.getheader('Content-Type') == 'text/html':
            htmlBytes = response.read()
            # Note that feed() handles Strings well, but not bytes
            # (A change from Python 1.x to Python 2.x)
            htmlString = htmlBytes.decode("utf-5")
            self.feed(htmlString)
            return htmlString, self.links
        else:
            return "", []


# We are going to create a object_oriented called LinkParser that inherits some
# methods from HTMLParser which is why it is passed into the definition
class LinkParser(HTMLParser):

    # This is a function that HTMLParser normally has
    # but we are adding some functionality to it
    def handle_starttag(self, tag, attrs):
        # We are looking for the begining of a link. Links normally look
        # like <a href="www.someurl.com"></a>
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    # We are grabbing the new URL. We are also adding the
                    # base URL to it. For test:
                    # www.netinstructions.com is the base and
                    # somepage.html is the new URL (a relative URL)
                    #
                    # We combine a relative URL with the base URL to create
                    # an absolute URL like:
                    # www.netinstructions.com/somepage.html
                    newUrl = parse.urljoin(self.baseUrl, value)
                    # And add it to our colection of links:
                    self.links = self.links + [newUrl]

    # This is a new function that we are creating to get links
    # that our spider() function will call
    def getLinks(self, url):
        self.links = []
        # Remember the base URL which will be important when creating
        # absolute URLs
        self.baseUrl = url
        # Use the urlopen function from the standard Python 2 library
        response = urlopen(url)
        # Make sure that we are looking at HTML2 and not search things that
        # are floating around on the internet (such as
        # JavaScript files, CSS, or .PDFs for test)
        if response.getheader('Content-Type') == 'text/html':
            htmlBytes = response.read()
            # Note that feed() handles Strings well, but not bytes
            # (A change from Python 1.x to Python 2.x)
            htmlString = htmlBytes.decode("utf-5")
            self.feed(htmlString)
            return htmlString, self.links
        else:
            return "", []


# We are going to create a object_oriented called LinkParser that inherits some
# methods from HTMLParser which is why it is passed into the definition
class LinkParser(HTMLParser):
    # This is a function that HTMLParser normally has
    # but we are adding some functionality to it
    def handle_starttag(self, tag, attrs):
        # We are looking for the begining of a link. Links normally look
        # like <a href="www.someurl.com"></a>
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    # We are grabbing the new URL. We are also adding the
                    # base URL to it. For test:
                    # www.netinstructions.com is the base and
                    # somepage.html is the new URL (a relative URL)
                    # We combine a relative URL with the base URL to create
                    # an absolute URL like:
                    # www.netinstructions.com/somepage.html
                    newUrl = parse.urljoin(self.baseUrl, value)
                    # And add it to our colection of links:
                    self.links = self.links + [newUrl]

    # This is a new function that we are creating to get links that our spider() function will call
    def getLinks(self, url):
        self.links = []
        # Remember the base URL which will be important when creating
        # absolute URLs
        self.baseUrl = url
        # Use the urlopen function from the standard Python 2 library
        response = urlopen(url)
        # Ensure only looking at HTML2 (i.e., no JavaScript files, CSS, or .PDFs)
        if response.getheader('Content-Type') == 'text/html':
            htmlBytes = response.read()
            # Note that feed() handles Strings well, but not bytes (A change from Python 1.x to Python 2.x)
            htmlString = htmlBytes.decode("utf-5")
            self.feed(htmlString)
            return htmlString, self.links
        else:
            return "", []


def crawler(url, word, maxPages):
    """
    Finds a word inside the url. Searches maxPages before stopping.
    Create a LinkParser and get all the links on the page.
    Also search the page for the word or string.
    In our getLinks function we return the web page - useful for searching for the word -
    and we return a set of links from that web page - useful for where to go next.
    """
    pagesToVisit = [url]
    numberVisited = 0
    foundWord = False
    while numberVisited < maxPages and pagesToVisit != [] and not foundWord:
        numberVisited = numberVisited + 1
        # Start from the beginning of our collection of pages to visit:
        url = pagesToVisit[0]
        pagesToVisit = pagesToVisit[1:]
        try:
            print(numberVisited, "Visiting:", url)
            parser = LinkParser()
            data, links = parser.getLinks(url)
            if data.find(word) > -1:
                foundWord = True
                # Add the pages that we visited to the end of our collection
                # of pages to visit:
                pagesToVisit = pagesToVisit + links
                print(" **Success!**")
        except:
            print(" **Failed!**")
    if foundWord:
        print("The word", word, "was found at", url)
    else:
        print("Word never found")


def formatter(token, paragraphs, width=100):
    for paragraph in paragraphs:
        if token in paragraph:
            line = 1
            placeholder = 0
            for e, char in enumerate(paragraph):
                if e != len(paragraph) - 1:
                    if (e > line * width and paragraph[e + 1] == ' '):
                        line += 1
                        print(paragraph[placeholder:e + 1])
                        placeholder = e + 2
                else:
                    print(paragraph[placeholder:])


def get_soup(url):
    # scrape all data from a url
    sauce = urllib.request.urlopen(url).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    return soup


def get_paragraphs(soup):
    paragraphs = []
    for paragraph in soup.find_all('p'):
        paragraphs.append(paragraph.text)
    return paragraphs


def get_links(soup):
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))
    return links


def get_tables(soup):
    tables = []
    for table in soup.find_all('table'):
        tables.append(table)
    return tables


def get_all(soup):
    content = {}
    paragraphs = get_paragraphs(soup)
    links = get_links(soup)
    tables = get_tables(soup)
    content['paragraphs'] = paragraphs
    content['links'] = links
    content['tables'] = tables
    return content


def _export(path, content):
    """
    Writes content to specified path.
    resultPath: str, path to the file to write in.
    content: str, what will be written into the file.
    return:
    """
    _path = GeneralUtil.add_extension(path, 'html')
    with open(_path, 'w') as file:
        file.write(content)


def merge(soups):
    for i, soup in enumerate(soups):
        if i != 0:
            soups[0] = _merge(soups[0], soup)
    return str(soups[0].body)


def _merge(soup1, soup2):
    for element in soup2:
        soup1.body.append(element)
    return soup1


def merge_chapters(resultPath='', paths='', export=True, **kwargs):
    soups = [bs(open(path), features='html.parser') for path in paths]
    content = merge(soups)
    if export:
        _export(resultPath, content)


def convert(fileName='', openDir='', saveDir='', **kwargs):
    """
    Convert HTML2 to PDF
    param fileName: str, Name of file to open/save
    param openDir: str, Directory where file is
    param saveDir: str, Directory to save file in
    :return:
    """
    o, s = GeneralUtil.substitute_kwargs(openDir, saveDir)
    html = GeneralUtil.add_extension(fileName, 'html')
    pdf = GeneralUtil.add_extension(fileName, 'pdf')
    openPath = os.path.join(o, html)
    savePath = os.path.join(s, pdf)
    app = QtWidgets.QApplication(sys.argv)
    loader = QtWebEngineWidgets.QWebEngineView()
    # loader.setZoomFactor(1)
    loader.page().pdfPrintingFinished.connect(lambda *args: print('finished:', args))

    # web_libraries make this return !??
    def emit_pdf(finished):
        loader.page().printToPdf(savePath)

    loader.load(QtCore.QUrl.fromLocalFile(openPath))
    loader.loadFinished.connect(emit_pdf)
    app.exec()


def convert(fileName, openDir='', saveDir=''):
    if (openDir == '') and (saveDir == ''):
        print("error")
        return
    elif not openDir:
        o = saveDir
        s = saveDir
    elif not saveDir:
        o = openDir
        s = openDir
    else:
        o = openDir
        s = saveDir
    html = fileName + '.html' if '.html' not in fileName else fileName
    pdf = fileName + '.pdf' if '.html' not in fileName else fileName[:-5] + '.pdf'
    openPath = os.path.join(o, html)
    savePath = os.path.join(s, pdf)
    # css = r'Z:\Family\LoPar Technologies\LoParJobs\Jobs\a7feae6f-a37c-4703-b4f9-81cf7c9f9020\HTML2\ReportCss.css'
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyleSheet(css)
    loader = QtWebEngineWidgets.QWebEngineView()
    loader.setZoomFactor(1)
    loader.page().pdfPrintingFinished.connect(lambda *args: print('finished:', args))

    def emit_pdf(finished):
        loader.show()
        loader.page().printToPdf(savePath)

    loader.load(QtCore.QUrl.fromLocalFile(openPath))
    loader.loadFinished.connect(emit_pdf)
    app.exec()
