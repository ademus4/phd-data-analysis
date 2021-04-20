import os
import luigi
import ROOT
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
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
    output_dir = luigi.Parameter(default=os.getenv('LUIGI_WORK_DIR'))
    config_file = luigi.Parameter()

    def output(self):
        _, fname = os.path.split(self.input_file)
        folder = fname.replace('.hipo', '')
        output_path = os.path.join(self.output_dir, folder)
        return luigi.LocalTarget(output_path)

    def requires(self):
        yield BuildFinalState(self.config_file)

    def run(self):
        ROOT.gROOT.ProcessLine(".x $CLAS12ROOT/RunRoot/LoadClas12Root.C");
        ROOT.gROOT.ProcessLine(".x $CHANSER/macros/Load.C")
        ROOT.gROOT.LoadMacro("$FINALSTATE/Pi2.cpp+")
        ROOT.gROOT.ProcessLine(".L $FINALSTATE/Run_Pi2.C")
        ROOT.Run_Pi2(self.input_file, self.config_file, self.output().path)
        

class ApplyCuts(luigi.Task):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=os.getenv('LUIGI_WORK_DIR'))
    topo = luigi.Parameter(default=0)

    def output(self):
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
            'adamt/Pi2_config__/FinalState.root'  # finalstate
        )
        data_tree = 'FINALOUTTREE'


        # load into dataframe ready for cuts
        df = ROOT.RDataFrame(data_tree, data_path)

        # simulated data has extra truth matching column, add this cut
        if "Truth" in df.GetColumnNames():
            df = df.Filter("Truth == 1")
        
        # apply the various kinematic cuts
        # define these in config file?
        df = df.Filter("Topo == 0")
        df = df.Filter("0.01 > Pi2MissMass2 > -0.01")
        df = df.Filter("0.5 > Pi2MissE > -0.5")
        df = df.Filter("Pi2MissP < 0.5")
        df = df.Filter("0.8 < Pi2MissMassnP < 1.1")
        df.Snapshot("withcuts", self.output().path)


class MomentFitting(luigi.Task):
    data_file = luigi.Parameter()
    sim_file = luigi.Parameter()
    nevents = luigi.Parameter()    
    mcmc = luigi.Parameter()
    output_dir = luigi.Parameter(default=os.getenv('LUIGI_WORK_DIR'))

    def requires(self):
        yield ApplyCuts(input_file=self.data_file), ApplyCuts(input_file=self.sim_file)

    def output(self):
        output_path = os.path.join(self.output_dir, 'moments/')
        return luigi.LocalTarget(output_path)

    def run(self):
        # save input conditions as txt file inside output folder?

        # make the output folder
        output_path = self.output().path
        os.mkdir(self.output().path)

        data_path = self.input()[0][0].path
        sim_path = self.input()[0][1].path
        data_tree = 'withcuts'

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


class MergeMoments(luigi.Task):
    data_file = luigi.Parameter()
    sim_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=os.getenv('LUIGI_WORK_DIR'))

    def requires(self):
        yield MomentFitting(data_file=self.data_file, sim_file=self.sim_file)

    def output(self):
        output_path = os.path.join(self.output_dir, 'moments', 'moments_merged.csv')
        return luigi.LocalTarget(output_path)

    def run(self):
        files = glob(os.path.join(self.input()[0].path, "**", "ResultsHS*.root"))

        # values related to moments fitting
        POL = 2
        L = 3
        M = 2

        # iterate over output files, load into dataframe, combine together
        dfs = []
        output = []
        for item in files:
            with uproot3.open(item) as data:

                # get bin name from path 
                path, _ = os.path.split(item)
                bin = float(path.split('/')[-1].replace('_', '').replace('Pi2MesonMass', ''))

                # use the results tree, same for both mcmc and minuit
                tree = data['ResultTree']

                # extract values from results tree
                for p in range(POL+1):
                    for l in range(L+1):
                        for m in range(M+1):
                            label = f"H{p}_{l}_{m}"
                            try:
                                val = tree[label].array()[0]
                                err = tree[label+'_err'].array()[0]
                            except KeyError:
                                # some combinations we can skip
                                continue

                            result = {
                                'val': val,
                                'err': err,
                                'p': p,
                                'l': l,
                                'm': m,
                                'bin': bin,
                                'label': label
                            }
                            output.append(result)

        # save output
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.output().path, index=False)


class PlotMoments(luigi.Task):
    data_file = luigi.Parameter()
    sim_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=os.getenv('LUIGI_WORK_DIR'))

    def requires(self):
        yield MergeMoments(data_file=self.data_file, sim_file=self.sim_file)

    def output(self):
        output_path = os.path.join(self.output_dir, 'moments', 'moments.png')
        return luigi.LocalTarget(output_path)

    def run(self):
        # get input file path
        input_file = self.input()[0].path

        df = pd.read_csv(input_file)
        labels = df['label'].unique()

        fig, axes = plt.subplots(4, 6, figsize=(30, 18))
        for i, ax in enumerate(axes.flatten()):
            if i>21:
                ax.set_axis_off()
                continue
            label = labels[i]
            data = df[df['label']==label].sort_values('bin')
            ax.errorbar(data['bin'], data['val'], xerr=None, yerr=data['err'], label=label)
            ax.set_ylim([-0.75, 0.75])
            ax.grid()
            ax.legend()

        fig.text(0.07, 0.5, "Moment mag", va='center', ha='center', rotation='vertical')
        fig.text(0.5, 0.05, "$\pi^+\pi^-$ Mass [GeV/$c^2$]", va='center', ha='center', )

        fig.savefig(self.output().path)


class Plotting(luigi.Task):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=os.getenv('LUIGI_WORK_DIR'))

    def output(self):
        prefix = self.input()[0][0].path.split('/')[-1]
        output_path = os.path.join(self.output_dir, prefix, 'plots')
        return luigi.LocalTarget(output_path)

    def requires(self):
        yield FinalState(input_file=self.input_file), ApplyCuts(input_file=self.input_file)

    def run(self):
        file_path_data = os.path.join(
            self.input()[0][0].path, 
            'adamt/Pi2_config__/FinalState.root'  # finalstate
        )
        file_path_cuts = self.input()[0][1].path

        # plot the raw data first
        output_dir = self.output().path
        A = Analysis(output_dir=output_dir)
        A.load_data(file_path_data, topo=0)
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
        A.load_data(file_path_cuts, topo=0)
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
        [PlotMoments()],
        workers=1, 
        local_scheduler=True
    )
