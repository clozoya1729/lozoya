from LoParDataProcessor import DataProcessor, DataOptimizer
from LoParDatasets.SyntheticData import SyntheticData

data = DataOptimizer.optimize_dataFrame(SyntheticData.data)
dtypes = DataProcessor.get_dtypes(data)
StaticPlotGenerator.generate_all_plots(data, dtypes)
