import os
import luigi
import ROOT
import matplotlib.pylab as plt
import numpy as np
import uproot3
import uproot4
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from luigi.contrib.external_program import (
    ExternalPythonProgramTask,
    ExternalProgramTask
)
from analysis import Analysis


class BuildFinalState(ExternalProgramTask):
    config_file = luigi.Parameter()

    def program_args(self):
        return [os.path.join(os.getenv('FINALSTATE'), "rebuild.sh")]

    def output(self):
        return luigi.LocalTarget(self.config_file)


class FinalState(luigi.Task):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter()
    config_file = luigi.Parameter()

    def output(self):
        _, fname = os.path.split(self.input_file)
        folder = fname.replace('.hipo', '')
        output_path = os.path.join(self.output_dir, folder)
        return luigi.LocalTarget(output_path)

    def requires(self):
        yield BuildFinalState(self.config_file)

    def run(self):
        ROOT.gROOT.ProcessLine(".x $CHANSER/macros/Load.C")
        ROOT.gROOT.LoadMacro("$FINALSTATE/Pi2.cpp+")
        ROOT.gROOT.ProcessLine(".L $FINALSTATE/Run_Pi2.C")
        ROOT.Run_Pi2(self.input_file, self.config_file, self.output().path)
        

class TestAnalysis(luigi.Task):
    input_file = luigi.Parameter()

    def output(self):
        prefix = self.input()[0].path.split('/')[-1]
        file_path = os.path.join('./plots/', prefix)
        return luigi.LocalTarget(file_path)

    def requires(self):
        yield FinalState(input_file=self.input_file)

    def run(self):
        file_path = os.path.join(
            self.input()[0].path, 
            'adamt/Pi2_config__/particleData/ParticleVariables_0.root'
        )
        A = Analysis(output_dir=self.output().path)
        A.load_data(file_path)

        os.mkdir(self.output().path)

        A.plot_exc_cuts()
        A.plot_timing()
        A.plot_mesons()
        A.plot_mesons(cuts=True)
        A.plot_meson_2D(cuts=True)
        A.plot_meson_decay_angle(cuts=True)


class ApplyCuts(luigi.Task):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter()

    def output(self):
        # something here like filename+cuts.root
        _, fname = os.path.split(self.input_file)
        folder = fname.replace('.hipo', '')
        output_path = os.path.join(self.output_dir, folder, 'data_cuts.root')
        return luigi.LocalTarget(output_path)

    def requires(self):
        yield FinalState(self.input_file)

    def run(self):
        # load finalstate root file using uproot3
        # filter data based on cut parameters
        # save in a seperate folder? output/skim3_******/filtered_data.root
        data_path = os.path.join(
            self.input()[0].path, 
            'adamt/Pi2_config__/FinalState.root'
        )
        data_tree = 'FINALOUTTREE'

        df = ROOT.RDataFrame(data_tree, data_path)
        df.Filter("-0.01 < Pi2MissMass2 < 0.01").Snapshot("FINALOUTTREE", self.output().path)
        #  .Filter("-0.5 < Pi2MissE < 0.5") \
        #  .Filter("Pi2MissP < 0.5") \
        #  .Filter("0.8 < Pi2MissMassnP < 1.1") \
          
          

class MomentFitting(luigi.Task):
    data_file = luigi.Parameter()
    sim_file = luigi.Parameter()
    nevents = luigi.Parameter()    
    mcmc = luigi.Parameter()

    def requires(self):
        yield ApplyCuts(input_file=self.data_file), ApplyCuts(input_file=self.sim_file)

    def output(self):
        file_path = os.path.join(os.getcwd(), 'output', 'moments')
        return luigi.LocalTarget(file_path)

    def run(self):
        # save input conditions as txt file inside output folder?

        # make the output folder
        output_path = self.output().path
        os.mkdir(self.output().path)

        # set the paths for brufit
        #data_path = os.path.join(
        #    self.input()[0][0].path, 
        #    'adamt/Pi2_config__/FinalState.root'
        #)
        #sim_path = os.path.join(
        #    self.input()[0][1].path, 
        #    'adamt/Pi2_config__/FinalState.root'
        #)
        data_path = self.input()[0][0].path
        sim_path = self.input()[0][1].path
        data_tree = 'Filtered'
        #sim_data_tree = 'Filtered'

        # load brufit root code
        ROOT.gROOT.ProcessLine(".x $BRUFIT/macros/LoadBru.C")
        ROOT.gROOT.ProcessLine(".L $MOMENTs/pshm_fit.C")

        # run the processing
        ROOT.pshm_fit(data_path,
                      data_tree,
                      sim_path,
                      data_tree,
                      output_path,
                      int(self.nevents),
                      int(self.mcmc))


class Plotting(luigi.Task):
    input_file = luigi.Parameter()

    def output(self):
        prefix = self.input()[0][0].path.split('/')[-1]
        file_path = os.path.join('./plots/', prefix)
        return luigi.LocalTarget(file_path)

    def requires(self):
        yield FinalState(input_file=self.input_file), ApplyCuts(input_file=self.input_file)

    def run(self):
        file_path_data = os.path.join(
            self.input()[0][0].path, 
            'adamt/Pi2_config__/particleData/ParticleVariables_0.root'
        )
        file_path_cuts = self.input()[0][1].path

        # plot the raw data first
        output_dir = self.output().path
        A = Analysis(output_dir=output_dir)
        A.load_data(file_path_data) # data_tree='FINALOUTTREE')
        A.plot_exc_cuts()
        A.plot_timing()
        A.plot_mesons()
        A.plot_electron()
        A.plot_proton()
        A.plot_pip()
        A.plot_pim()

        # plots the data with cuts applied
        output_dir = os.path.join(self.output().path, 'cuts')
        A = Analysis(output_dir=output_dir)
        A.load_data(file_path_cuts) # data_tree='FINALOUTTREE')
        A.plot_mesons()
        A.plot_meson_2D()
        A.plot_meson_decay_angle()


class BulkFinalState(luigi.WrapperTask):
    input_path = luigi.Parameter()

    def requires(self):
        # get a list of hipo files in the path and submit them as final state tasks
        input_files = glob(self.input_path)
        for item in input_files:
            yield FinalState(item)


if __name__ == '__main__':
    luigi.build(
        [Plotting(input_file='/data/sims/job_2675.hipo')],
        workers=1, 
        local_scheduler=True
    )
