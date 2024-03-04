import pandas as pd
import os
import configuration
import lozoya.file
import lozoya.gui
import lozoya.ml


class ConfigurationDataFrame:
    def __init__(self):
        self.dataFrames = [pd.DataFrame()]


class App(lozoya.gui.TSApp):
    def __init__(self, name, root):
        lozoya.gui.TSApp.__init__(self, name, root)
        self.setStyleSheet(lozoya.gui.qss)
        self.setup_classification()
        self.setup_clustering()
        # self.setup_dataframes()
        self.setup_datasets()
        self.setup_decomposition()
        self.setup_preprocessing()
        self.setup_regression()
        self.formSwitcher = lozoya.gui.TSListWidgetSwitcher(
            widgets=[
                self.formSwitcherDataset,
                self.formSwitcherSamplesGenerator,
                self.formSwitcherPreprocessing,
                self.formSwitcherDecomposition,
                self.formSwitcherClassification,
                self.formSwitcherClustering,
                self.formSwitcherRegression,
            ]
        )
        self.plot = lozoya.gui.TSPlot(self, width=5, height=4, dpi=100)
        self.formSwitcher.add_field(self.plot)
        # self.formSwitcher.add_field(self.formSwitcherDataFrames)
        self.window.setCentralWidget(self.formSwitcher)

    def add_dataframe(self):
        self.configuratorDataFrame.dataFrames.append(pd.DataFrame())
        index = self.configuratorDataFrame.dataFrames.__len__()
        path = os.path.join(self.root, f'dataframes{str(index)}')
        form = lozoya.gui.make_form_from_class(
            template=self.configuratorDataFrame.dataFrames[index],
        )
        self.formsDataFrames.append(form)
        self.formSwitcherDataFrames.add_field(form)

    def load_data_from_sklearn_datasets(self):
        indexDataset = self.formSwitcherDataset.currentIndex
        values = self.formSwitcherDataset.currentForm.get_values()
        data = lozoya.ml.SklearnModels.dataset[indexDataset](**values)
        self.configuratorDataFrame.dataframes[indexDataset] = data['data']
        indexDataFrame = self.formSwitcherDataFrames.currentIndex
        self.plot.plot(data=self.configuratorDataFrame.dataFrames[indexDataFrame])

    def load_data_from_sklearn_samples_generator(self):
        index = self.formSwitcherSamplesGenerator.currentIndex
        values = self.formSwitcherSamplesGenerator.currentForm.get_values()
        data = lozoya.ml.SklearnModels.samplesGenerator[index](**values)
        self.plot.plot(x=data[0][:, 0], y=data[0][:, 1], color=data[1])

    def remove_dataframe(self, index):
        del self.dataFrames[index]

    def setup_classification(self):
        self.configuratorsClassification = lozoya.file.make_configuration_files_from_class(
            templates=lozoya.ml.SklearnModels.classification,
            root=self.root,
        )
        self.formsClassification = lozoya.gui.make_forms_from_class(
            templates=self.configuratorsClassification
        )
        self.formSwitcherClassification = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsClassification,
            name='Classification',
        )

    def setup_clustering(self):
        self.configuratorsClustering = lozoya.file.make_configuration_files_from_class(
            templates=lozoya.ml.SklearnModels.clustering,
            root=self.root,
        )
        self.formsClustering = lozoya.gui.make_forms_from_class(
            templates=self.configuratorsClustering,
        )
        self.formSwitcherClustering = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsClustering,
            name='Clustering',
        )

    def setup_datasets(self):
        self.configuratorsDataset = lozoya.file.make_configuration_files_from_function(
            templates=lozoya.ml.SklearnModels.dataset,
            root=self.root,
        )
        self.formsDataset = lozoya.gui.make_forms_from_class(
            templates=self.configuratorsDataset
        )
        self.formSwitcherDataset = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsDataset,
            name='Dataset',
        )
        self.configuratorsSamplesGenerator = lozoya.file.make_configuration_files_from_function(
            templates=lozoya.ml.SklearnModels.samplesGenerator,
            root=self.root,
        )
        self.formsSamplesGenerator = lozoya.gui.make_forms_from_class(
            templates=self.configuratorsSamplesGenerator
        )
        self.formSwitcherSamplesGenerator = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsSamplesGenerator,
            name='Samples Generator',
        )
        buttonA = lozoya.gui.TSInputButton(
            name='',
            text='Load',
            callback=self.load_data_from_sklearn_datasets,
        )
        self.formSwitcherDataset.add_field(buttonA)
        buttonB = lozoya.gui.TSInputButton(
            name='',
            text='Load',
            callback=self.load_data_from_sklearn_samples_generator,
        )
        self.formSwitcherSamplesGenerator.add_field(buttonB)

    def setup_dataframes(self):
        self.configuratorDataFrame = lozoya.file.make_configuration_file_from_class(
            template=ConfigurationDataFrame,
            path=os.path.join(self.root, 'dataframes'),
        )
        for i, dataframe in enumerate(self.configuratorDataFrame.dataFrames):
            dataframe.__name__ = f'dataframe{i}'
        self.formsDataFrames = lozoya.gui.make_forms_from_class(
            templates=self.configuratorDataFrame.dataFrames,
        )
        self.formSwitcherDataFrames = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsDataFrames,
            name='DataFrames',
        )
        self.buttonAddDataframe = lozoya.gui.TSInputButton(
            name='Add Dataframe',
            text='Add Dataframe',
            callback=self.add_dataframe,
        )
        self.formSwitcherDataFrames.add_field(self.buttonAddDataframe)

    def setup_decomposition(self):
        self.configuratorsDecomposition = lozoya.file.make_configuration_files_from_function(
            templates=lozoya.ml.SklearnModels.decomposition,
            root=self.root,
        )
        self.formsDecomposition = lozoya.gui.make_forms_from_class(
            templates=self.configuratorsDecomposition
        )
        self.formSwitcherDecomposition = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsDecomposition,
            name='Decomposition',
        )

    def setup_preprocessing(self):
        self.configuratorsPreprocessing = lozoya.file.make_configuration_files_from_class(
            templates=lozoya.ml.SklearnModels.preprocessing,
            root=self.root,
        )
        self.formsPreprocessing = lozoya.gui.make_forms_from_class(
            templates=self.configuratorsPreprocessing,
        )
        self.formSwitcherPreprocessing = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsPreprocessing,
            name='Preprocessing',
        )

    def setup_regression(self):
        self.configuratorsRegression = lozoya.file.make_configuration_files_from_class(
            templates=lozoya.ml.SklearnModels.regression,
            root=self.root,
        )
        self.formsRegression = lozoya.gui.make_forms_from_class(
            templates=self.configuratorsRegression
        )
        self.formSwitcherRegression = lozoya.gui.TSListWidgetSwitcher(
            widgets=self.formsRegression,
            name='Regression',
        )


App(
    name=configuration.name,
    root=configuration.root,
).exec()
