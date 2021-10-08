import corner
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


class DefaultParams(luigi.Config):
    finalstate_dir = luigi.Parameter()
    config_file = luigi.Parameter()
    output_dir = luigi.Parameter()
    topo = luigi.Parameter()
    POL = luigi.IntParameter()
    L = luigi.IntParameter()
    M = luigi.IntParameter()


class BuildFinalState(ExternalProgramTask):
    config_file = luigi.Parameter(default=DefaultParams().config_file)

    def program_args(self):
        return [os.path.join(DefaultParams().finalstate_dir, "rebuild.sh")]

    def output(self):
        return luigi.LocalTarget(self.config_file)


class FinalState(ExternalPythonProgramTask):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)
    config_file = luigi.Parameter(default=DefaultParams().config_file)
    
    def requires(self):
        yield BuildFinalState(self.config_file)

    def output(self):
        _, fname = os.path.split(self.input_file)
        folder = fname.replace('.hipo', '')
        output_path = os.path.join(self.output_dir, folder)
        return luigi.LocalTarget(output_path)

    def program_args(self):
        command = [
            'python', 'finalstate/run_pi2.py',
            '-i', self.input_file,
            '-o', self.output().path,
            '-c', self.config_file
        ]
        return command


class ApplyCuts(luigi.Task):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)
    topo = luigi.Parameter(default=DefaultParams().topo)

    def output(self):
        _, fname = os.path.split(self.input_file)
        folder = fname.replace('.hipo', '')
        output_path = os.path.join(self.output_dir, folder, 'data_cuts.root')
        return luigi.LocalTarget(output_path)

    def requires(self):
        yield FinalState(self.input_file, self.output_dir)

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
        df = df.Filter("Topo == 0") \
            .Filter("Pi2MissMass2 > -0.01").Filter("Pi2MissMass2 < 0.01") \
            .Filter("Pi2MissE > -0.5").Filter("Pi2MissE < 0.5") \
            .Filter("Pi2MissP < 0.5") \
            .Filter("Pi2MissMassnP > 0.8").Filter("Pi2MissMassnP < 1.1") \
            .Filter("Pi2ElP > 0.5").Filter("Pi2ElP < 4.5")
        df.Snapshot("withcuts", self.output().path)


class MergeData(ExternalProgramTask):
    input_files = luigi.Parameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)
    output_filename = luigi.Parameter(default='merged.root')

    def program_args(self):
        return ["hadd", self.output().path] + glob(self.input_files)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_dir,
                                              self.output_filename))


class Plotting(luigi.Task):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)

    def output(self):
        prefix = self.input()[0][0].path.split('/')[-1]
        output_path = os.path.join(self.output_dir, prefix, 'plots')
        return luigi.LocalTarget(output_path)

    def requires(self):
        yield [
            FinalState(input_file=self.input_file, output_dir=self.output_dir), 
            ApplyCuts(input_file=self.input_file, output_dir=self.output_dir)
        ]

    def run(self):
        file_path_data = os.path.join(
            self.input()[0][0].path, 
            'adamt/Pi2_config__/FinalState.root'  # finalstate
        )
        file_path_cuts = self.input()[0][1].path

        # plot the raw data first
        output_dir = self.output().path
        A = Analysis(output_dir=output_dir)
        A.load_data(file_path_data, tree='FINALOUTTREE', label='data', topo=0)
        A.plot_exc_cuts()
        A.plot_mesons()
        A.plot_meson_2D()
        A.plot_meson_decay_angle()
        A.plot_meson_decay_angle_1D()
        A.plot_electron()
        A.plot_proton()
        A.plot_pip()
        A.plot_pim()

        # plots the data with cuts applied
        output_dir = os.path.join(self.output().path, 'cuts')
        A = Analysis(output_dir=output_dir)
        A.load_data(file_path_cuts, tree='withcuts', label='data', topo=0)
        A.plot_exc_cuts()
        A.plot_mesons()
        A.plot_meson_2D()
        A.plot_meson_decay_angle()
        A.plot_meson_decay_angle_1D()
        A.plot_electron()
        A.plot_proton()
        A.plot_pip()
        A.plot_pim()


class PlotFile(luigi.Task):
    input_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)

    def output(self):
        filepath, filename = os.path.split(self.input_file)
        output_path = os.path.join(filepath, filename.split('.')[0])
        return luigi.LocalTarget(output_path)

    def run(self):

        # plots
        output_dir = self.output().path
        A = Analysis(output_dir=output_dir)
        A.load_data(self.input_file, topo=0)
        A.plot_exc_cuts()
        A.plot_timing()
        A.plot_mesons()
        A.plot_electron()
        A.plot_proton()
        A.plot_pip()
        A.plot_pim()
        A.plot_meson_2D()
        A.plot_meson_decay_angle()


class PlotFilesComparison(luigi.Task):
    input_file1 = luigi.Parameter()
    label1 = luigi.Parameter()
    tree1 = luigi.Parameter()
    input_file2 = luigi.Parameter()
    label2 = luigi.Parameter()
    tree2 = luigi.Parameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)

    def output(self):
        output_path = os.path.join(self.output_dir, 'comparison')
        return luigi.LocalTarget(output_path)

    def run(self):

        # plots
        output_dir = self.output().path
        A = Analysis(output_dir=output_dir)
        A.load_data(self.input_file1, self.tree1, self.label1, topo=0)
        A.load_data(self.input_file2, self.tree2, self.label2, topo=0)
        A.plot_exc_cuts(density=True)
        A.plot_mesons(density=True)
        A.plot_electron(density=True)
        A.plot_proton()
        A.plot_pip()
        A.plot_pim()
        A.plot_meson_decay_angle_1D(density=True)


class MomentFitting(luigi.Task):
    # check for folders, not for tasks
    data_file = luigi.Parameter()
    data_tree = luigi.Parameter()
    sim_file = luigi.Parameter()
    sim_tree = luigi.Parameter()
    bins = luigi.IntParameter()
    bmin = luigi.FloatParameter()
    bmax = luigi.FloatParameter()
    nevents = luigi.IntParameter()    
    mcmc = luigi.IntParameter()
    L = luigi.IntParameter(default=DefaultParams().L)
    M = luigi.IntParameter(default=DefaultParams().M)
    nCores = luigi.IntParameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)

    def output(self):
        output_path = os.path.join(self.output_dir, 'moments/')
        return luigi.LocalTarget(output_path)

    def run(self):
        # make the output folder
        output_path = self.output().path
        os.mkdir(self.output().path)

        # load brufit root code
        ROOT.gROOT.ProcessLine(".x $BRUFIT/macros/LoadBru.C")
        ROOT.gROOT.ProcessLine(".L $MOMENTs/pshm_fit.C")

        params = {
            "data_path": self.data_file,
            "data_tree": self.data_tree,
            "sim_data_path": self.sim_file,
            "sim_data_tree": self.sim_tree,
            "output_path": output_path,
            "bins": self.bins,
            "bmin": self.bmin,
            "bmax": self.bmax,
            "nEvents": self.nevents,
            "mcmc": self.mcmc,
            "L": self.L,
            "M": self.M,
            "nCores": self.nCores
        }

        # run the processing
        ROOT.pshm_fit(**params)


class MergeMoments(luigi.Task):
    data_file = luigi.Parameter()
    sim_file = luigi.Parameter()
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)
    POL = luigi.Parameter(default=DefaultParams().POL)
    L = luigi.Parameter(default=DefaultParams().L)
    M = luigi.Parameter(default=DefaultParams().M)

    def requires(self):
        yield MomentFitting(data_file=self.data_file, sim_file=self.sim_file)

    def output(self):
        output_path = os.path.join(self.output_dir, 'moments', 'moments_merged.csv')
        return luigi.LocalTarget(output_path)

    def run(self):
        files = glob(os.path.join(self.input()[0].path, "**", "ResultsHS*.root"))

        # values related to moments fitting
        POL = self.POL
        L = self.L
        M = self.M

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
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)

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
            ax.errorbar(data['bin'], data['val'], xerr=None, yerr=data['err'], label=label, marker='.', ls='none')
            ax.set_ylim([-0.75, 0.75])
            ax.grid()
            ax.legend()

        fig.text(0.07, 0.5, "Moment mag", va='center', ha='center', rotation='vertical')
        fig.text(0.5, 0.05, "$\pi^+\pi^-$ Mass [GeV/$c^2$]", va='center', ha='center', )

        fig.savefig(self.output().path)


class MCMCMomentsPlotsPerBin(luigi.Task):
    input_file = luigi.Parameter()
    burnin = luigi.Parameter(default=5000)
    output_dir = luigi.Parameter(default=DefaultParams().output_dir)
    POL = luigi.Parameter(default=DefaultParams().POL)
    L = luigi.Parameter(default=DefaultParams().L)
    M = luigi.Parameter(default=DefaultParams().M)

    def output(self):
        # folder for the individual bin
        path, _ = os.path.split(self.input_file)
        bin = float(path.split('/')[-1].replace('_', '').replace('Pi2MesonMass', ''))
        output_path = os.path.join(self.output_dir, 'moments', 'plots', f'bin_{bin}')
        return luigi.LocalTarget(output_path)

    def run(self):

        # values related to moments fitting
        POL = self.POL
        L = self.L
        M = self.M

        yields = []
        values = []
        labels = []
        with uproot3.open(self.input_file) as data:

            # get bin name from path 
            path, _ = os.path.split(self.input_file)
            bin = float(path.split('/')[-1].replace('_', '').replace('Pi2MesonMass', ''))
            print(f'Corner plots for bin {bin}')

            # make folder for plots
            os.makedirs(self.output().path)

            # use mcmctree for raw info on fit
            tree = data['MCMCTree']

            yields.append({
                'bin': bin,
                'yield': tree['Yld_Moments'].array()
            })


            # extract values from mcmc tree
            for p in range(POL+1):
                for l in range(L+1):
                    for m in range(M+1):
                        label = f"H{p}_{l}_{m}"
                        try:
                            vals = tree[label].array()
                        except KeyError:
                            # some combinations we can skip
                            continue
                        
                        values.append(vals)
                        labels.append(label)

        # corner plot
        fig = corner.corner(
            np.array(values).T, 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, title_kwargs={"fontsize": 12})

        filename = os.path.join(self.output().path, 'corner_plot.png')
        fig.savefig(filename)
        fig.clear()
        plt.close(fig)

        # another corner plot after burn in
        fig = corner.corner(
            np.array(values).T[self.burnin:], 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, title_kwargs={"fontsize": 12})

        filename = os.path.join(self.output().path, 'corner_plot_burnin.png')
        fig.savefig(filename)
        fig.clear()
        plt.close(fig)

        # plot the timelines
        N = len(labels)
        fig, axes = plt.subplots(int(N/4)+1, 4, figsize=(20, 20))
        axes = axes.flatten()
        i = 0
        for label, vals in zip(labels, values):            
            ax = axes[i]
            if i>=N:
                ax.remove()
                continue
            ax.plot(vals, label=label)
            ax.legend()
            ax.set_xlabel('events')
            ax.grid()
            i += 1
        
        filename = os.path.join(self.output().path, 'timeline_plots.png')
        plt.tight_layout()
        fig.savefig(filename)
        fig.clear()
        plt.close(fig)
