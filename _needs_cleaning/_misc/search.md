# Search Engine

The search engine module serves the following purposes:
<ul>
    <li><a href="#C1">Extract Data</a></li>
    <li><a href="#C2">Prepare Data</a></li>
    <li><a href="#C3">Reduce Data</a></li>
    <li><a href="#C4">Tabulate Data</a></li>
</ul>

## Extract Data

The search engine reads all files of format .csv, .dat, .hdf, .html, .txt, .xlsx
and provides an interface to query the data within them. The interface is automatically
populated with a search field into which queries may be entered as comma separated lists.
Each entry in the search field corresponds directly to a column of the data. The search
engine creates a copy of the data and removes from it all the rows which do not contain
a match to the query. The resulting data is presented in tables. There is one table per
data file containing a query match and the tables are separated by tabs. The results may
also be exported in the aforementioned formats.

## Prepare Data

The results of a query can be saved to a specified folder to be used in any of the
software's other modules, such as Machine Learning, Simulation, etc.

## Reduce Data

Each entry in the search field is accompanied by a check box. When an entry's check
box is checked, the data in the column corresponding to the entry will be included in
the results of the query. If the check box is not checked, the data will not appear in
the results and the query will process faster.

## Tabulate Data<

The results are presented in publication-ready and interactive tables.
These tables are html files which can be embedded or saved as .png.