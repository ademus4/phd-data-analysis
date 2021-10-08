import os
import matplotlib.pylab as plt
import numpy as np
import uproot3
import uproot4
from concurrent.futures import ThreadPoolExecutor

DETECTOR_REGIONS = {
    1000: 'FT',
    2000: 'FD',
    3000: 'CD'
}


class Analysis:
    def __init__(self, output_dir, n_workers=4, cache="4 GB"):
        self.output_dir = output_dir

        self.cut_mm2=0.01
        self.cut_mE=0.5
        self.cut_mP=0.5
        self.cut_mm2pi=(0.8, 1.1)
        self.cut_elp=(1, 4.5)
        self.tree_data = []
        self.tree_data_cut = []
        self.datasets = {}

        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        self.cache = uproot4.LRUArrayCache(cache)

        plt.style.use('seaborn-whitegrid')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.formatter.limits'] = (-3,3)
        plt.rcParams['image.cmap'] = 'jet'
        plt.rcParams['font.size']= 22
        plt.rcParams['axes.labelsize'] = 22
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['figure.figsize'] = 8, 6
        plt.rcParams['legend.fontsize']= 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

    def load_data(self, path, tree, label, topo=None):
        if len(self.datasets) == 2:
            raise ValueError('Cannot compare more than 2 datasets')

        print("Loading:")
        print(path)
        tree_data = uproot4.lazy({path: tree},
                                 executor=self.executor,
                                 blocking=False,
                                 cache=self.cache)
        
        # check if topo given, careful with zero!
        if topo is not None:
            tree_data = tree_data[tree_data['Topo'] == topo]

        self.datasets[label] = tree_data

    def apply_cuts(self):
        cuts = (
            #np.array(self.tree_data['Pi2TriggerMesonex']==1) & 
            np.array(np.abs(self.tree_data['Pi2MissMass2'])<self.cut_mm2) &
            np.array(np.abs(self.tree_data['Pi2MissE'])<self.cut_mE) &
            np.array(self.tree_data['Pi2MissP']<self.cut_mP) &
            np.array(
                (self.cut_mm2pi[0]<self.tree_data['Pi2MissMassnP']) & 
                (self.tree_data['Pi2MissMassnP']<self.cut_mm2pi[1]))
        )
        self.tree_data_cut = self.tree_data[cuts]

    def plot_exc_cuts(self, density=False):
        params = {
            'bins': 201,
            'histtype': 'step',
            'linewidth': 2
        }

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for i, (label, data) in enumerate(self.datasets.items()):
            n = 0
            h = axes[n].hist(np.array(data['Pi2MissMass2']),
                             range=(-0.05, 0.05), label=label, density=density,
                             **params)
            axes[n].axvline(-self.cut_mm2, color='red')
            axes[n].axvline(self.cut_mm2, color='red')
            axes[n].set_ylabel('Events')
            axes[n].set_xlabel('Missing Mass ^2')

            n += 1
            h = axes[n].hist(np.array(data['Pi2MissE']),
                             range=(-2, 2), label=label, density=density,
                             **params)
            axes[n].axvline(-self.cut_mE, color='red')
            axes[n].axvline(self.cut_mE, color='red')
            axes[n].set_xlabel('Missing Energy')

            n += 1
            h = axes[n].hist(np.array(data['Pi2MissP']),
                             range=(-0.1, 1), label=label, density=density,
                             **params)
            axes[n].axvline(self.cut_mP, color='red')
            axes[n].set_xlabel('Missing Momentum')

            n += 1
            h = axes[n].hist(np.array(data['Pi2MissMassnP']),
                             range=(0, 2), label=label, density=density,
                             **params)
            axes[n].set_xlabel('Missing Mass 2pi')
            axes[n].axvline(self.cut_mm2pi[0], color='red')
            axes[n].axvline(self.cut_mm2pi[1], color='red')

            n += 1
            h = axes[n].hist(np.array(data['Pi2ElP']),
                             range=(0, 6), label=label, density=density,
                             **params)
            axes[n].set_xlabel('Electron Momentum [GeV]')
            axes[n].axvline(self.cut_elp[0], color='red')
            axes[n].axvline(self.cut_elp[1], color='red')

            # add legends to plots if they use more than 1 dataset
            if len(self.datasets) > 1:
                for v in range(n+1):
                    axes[v].legend()

        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir, 'exclusivity_cuts.png'))

    def plot_timing(self, xrange=(-0.5, 0.5)):
        vals = [
            "Pi2ElTime",
            "Pi2ProtTime",
            "Pi2PipTime",
            "Pi2PimTime"
        ]

        params = {
            'range': xrange,
            'bins': 101,
            'histtype': 'step',
            'linewidth': 2
        }

        fig, ax = plt.subplots()
        for i, val in enumerate(vals):
            ax.hist(np.array(self.tree_data[val]), label=val, **params)
        ax.legend()
        ax.set_ylabel('Events (per bin)')
        ax.set_xlabel('Time [ns]')
        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir, 'timing_plots.png'))

    def plot_mesons(self, density=False):
        params = {
            'bins': 201,
            'histtype': 'step',
            'linewidth': 2
        }

        vals = [
            ["Pi2MesonMass", (0, 3), 'Mass [GeV/$c^2$]'],
            ["Pi2DppMass",   (1, 4), 'Mass [GeV/$c^2$]'],
            ["Pi2D0Mass",    (1, 4), 'Mass [GeV/$c^2$]'],
            ["Pi2t",         (0, 5), '-t [GeV/c]'],
        ]

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        axes = ax.flatten()
        for label, data in self.datasets.items():
            for i, valr in enumerate(vals):
                val, r , xlab = valr
                h = axes[i].hist(np.abs(np.array(data[val])),
                                 range=r, label=label, density=density,
                                 **params)
                axes[i].set_title(val.replace('Pi2', ''))
                axes[i].set_ylabel('Events (per bin)')
                axes[i].set_xlabel(xlab)

                # add legends to plots if they use more than 1 dataset
                if len(self.datasets) > 1:
                    axes[i].legend()

        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir, 'meson_plots.png'))

    def plot_electron(self, density=False):
        params = {
            'histtype': 'step',
            'linewidth': 2
        }

        vals = [
            ["Pi2ElP",      (0, 7), 201,   'Momentum [GeV/c]'],
            ["Pi2ElTh",     (1, 7), 201,   '$\\theta$ [deg]'],
            ["Pi2ElDE",     (0, 30), 201,  'Delta Energy [MeV?]'],
            ["Pi2ElRegion", (0, 5000), 21, 'Region'],
            ["Pi2Egamma",   (0, 11), 201,  'Energy [GeV]'],
        ]

        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        axes = ax.flatten()
        for label, data in self.datasets.items():
            for i, valr in enumerate(vals):
                val, r, bins, xlab = valr
                if val[-2:] == 'Th':
                    norm = 180/np.pi
                else:
                    norm = 1
                h = axes[i].hist(np.array(data[val])*norm,
                                 range=r, bins=bins, label=label,
                                 density=density, **params)
                axes[i].set_title(val.replace('Pi2', ''))
                axes[i].set_ylabel('Events (per bin)')
                axes[i].set_xlabel(xlab)

                # add legends to plots if they use more than 1 dataset
                if len(self.datasets) > 1:
                    axes[i].legend()

        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir, 'electron_plots.png'))

    def plot_proton(self, density=False):
        linestyles = ['-', '--']
        linecolours = ['b', 'orange', 'g']
        params = {
            'histtype': 'step',
            'linewidth': 2
        }
        vals = [
            ["Pi2ProtP",      (0, 7),   201, 'Momentum [GeV/c]'],
            ["Pi2ProtTh",     (0, 110), 201, '$\\theta$ [deg]'],
            ["Pi2ProtRegion", (0, 5000), 21, 'Region'],
        ]

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        axes = ax.flatten()
        for j, (_, data) in enumerate(self.datasets.items()):
            for i, valr in enumerate(vals):
                val, r, bins, xlab = valr
                if val[-2:] == 'Th':
                    norm = 180/np.pi
                else:
                    norm = 1
                for c, (region, label) in enumerate(DETECTOR_REGIONS.items()):
                    h = axes[i].hist(
                        np.array(data[data['Pi2ProtRegion']==region][val])*norm,
                        range=r, bins=bins, label=label, density=density,
                        ec=linecolours[c], linestyle=linestyles[j], **params)
                axes[i].set_title(val.replace('Pi2', ''))
                axes[i].set_ylabel('Events (per bin)')
                axes[i].set_xlabel(xlab)
                if j == 0:
                    axes[i].legend()

        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir,  'proton_plots.png'))
        plt.close('all')

    def plot_pip(self, density=False):
        linestyles = ['-', '--']
        linecolours = ['b', 'orange', 'g']
        params = {
            'histtype': 'step',
            'linewidth': 2
        }
        vals = [
            ["Pi2PipP",      (0, 7), 201,      'Momentum [GeV/c]'],
            ["Pi2PipTh",     (0, 110), 201,    '$\\theta$ [deg]'],
            ["Pi2PipRegion", (0, 5000), 21,    'Region'],
        ]

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        axes = ax.flatten()
        for j, (_, data) in enumerate(self.datasets.items()):
            for i, valr in enumerate(vals):
                val, r, bins, xlab = valr
                if val[-2:] == 'Th':
                    norm = 180/np.pi
                else:
                    norm = 1
                for c, (region, label) in enumerate(DETECTOR_REGIONS.items()):
                    h = axes[i].hist(
                        np.array(data[data['Pi2PipRegion']==region][val])*norm,
                        range=r, bins=bins, label=label, density=density,
                        ec=linecolours[c], linestyle=linestyles[j], **params)
                axes[i].set_title(val.replace('Pi2', ''))
                axes[i].set_ylabel('Events (per bin)')
                axes[i].set_xlabel(xlab)
                if j == 0:
                    axes[i].legend()

        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir, 'pip_plots.png'))
        plt.close('all')

    def plot_pim(self, density=False):
        linestyles = ['-', '--']
        linecolours = ['b', 'orange', 'g']
        params = {
            'histtype': 'step',
            'linewidth': 2
        }
        vals = [
            ["Pi2PimP",      (0, 7), 201,   'Momentum [GeV/c]'],
            ["Pi2PimTh",     (0, 80), 201,  '$\\theta$ [deg]'],
            ["Pi2PimRegion", (0, 5000), 21, 'Region'],
        ]

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        axes = ax.flatten()
        for j, (_, data) in enumerate(self.datasets.items()):
            for i, valr in enumerate(vals):
                val, r, bins, xlab = valr
                if val[-2:] == 'Th':
                    norm = 180/np.pi
                else:
                    norm = 1
                for c, (region, label) in enumerate(DETECTOR_REGIONS.items()):
                    h = axes[i].hist(
                        np.array(data[data['Pi2PimRegion']==region][val])*norm,
                        range=r, bins=bins, label=label, density=density,
                        ec=linecolours[c], linestyle=linestyles[j], **params)
                axes[i].set_title(val.replace('Pi2', ''))
                axes[i].set_ylabel('Events (per bin)')
                axes[i].set_xlabel(xlab)
                if j == 0:
                    axes[i].legend()

        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir, 'pim_plots.png'))
        plt.close('all')

    def plot_meson_2D(self):
        val1 = ["Pi2MesonMass", "2$\pi$ Mass"]
        val2 = ["Pi2D0Mass",    "$\Delta$0 Mass"]
        val3 = ["Pi2DppMass",   "$\Delta$++ Mass"]

        params = {
            'range': [[0.5, 2.5], [1, 3.5]],
            'bins': 100,
        }

        _, data = list(self.datasets.items())[0]

        fig, axes = plt.subplots(1, 2, figsize=(16,6))
        axes[0].hist2d(
            np.array(data[val1[0]]), 
            np.array(data[val2[0]]),
            **params)
        axes[0].set_xlabel(val1[1])
        axes[0].set_ylabel(val2[1])

        axes[1].hist2d(
            np.array(data[val1[0]]), 
            np.array(data[val3[0]]),
            **params)
        axes[1].set_xlabel(val1[1])
        axes[1].set_ylabel(val3[1])

        plt.tight_layout()

        # create output dir if it doesnt exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        fig.savefig(os.path.join(self.output_dir, 'meson_2D_plots.png'))

    def plot_meson_decay_angle(self):
        plots = [
            {
                'filename': 'meson_decay_GJ_phi.png',
                'vals': ['Pi2MesonMass', 'Pi2MesonGJPhi'],
                'labels': ['Meson Mass [GeV/$c^2$]', 'Meson Decay Angle (GJ) Phi'],
                'params': {
                   'range': [[0, 3.5], [-3.5, 3.5]],
                    'bins': 100,
                }
            },
            {
                'filename': 'meson_decay_GJ_costh.png',
                'vals': ['Pi2MesonMass', 'Pi2MesonGJCosTh'],
                'labels': ['Meson Mass [GeV/$c^2$]', 'Meson Decay Angle (GJ) CosTh'],
                'params': {
                   'range': [[0, 3.5], [-1, 1]],
                    'bins': 100,
                }
            },
            {
                'filename': 'meson_decay_H_phi.png',
                'vals': ['Pi2MesonMass', 'Pi2MesonHPhi'],
                'labels': ['Meson Mass [GeV/$c^2$]', 'Meson Decay Angle (helicity) Phi'],
                'params': {
                    'range': [[0, 3.5], [-3.5, 3.5]],
                    'bins': 100,
                }
            },
            {
                'filename': 'meson_decay_H_costh.png',
                'vals': ['Pi2MesonMass', 'Pi2MesonHCosTh'],
                'labels': ['Meson Mass [GeV/$c^2$]', 'Meson Decay Angle (helicity) CosTh'],
                'params': {
                    'range': [[0, 3.5], [-1, 1]],
                    'bins': 100,
                }
            }
        ]

        # mass bins for meson
        mbwidth = 0.2
        mbins = np.arange(0, 2, mbwidth)

        font = {
            'family': 'serif',
            'color':  'white',
            'weight': 'normal',
            'size': 14,
        }

        _, data = list(self.datasets.items())[0]

        for plot in plots:
            params = plot['params']
            vals = plot['vals']
            labels = plot['labels']
            fig, axes = plt.subplots(3, 3, 
                                    figsize=(12, 10), 
                                    sharex=True, sharey=True, 
                                    gridspec_kw={'hspace': 0, 'wspace': 0})
            for i, ax in enumerate(axes.flatten()):
                if i>=len(mbins):
                    ax.remove()
                    continue
                mass_low = mbins[i]
                mass_high = mass_low + mbwidth
                d = data[np.array(-data['Pi2t']>=mass_low) & np.array(-data['Pi2t']<=mass_high)]
                x, y = [np.array(d[vals[0]]), np.array(d[vals[1]])]
                ax.hist2d(x, y, **params)
                label = '{:.2f}>-t>{:.2f}'.format(mass_low, mass_high)
                ax.text(
                    0.5, 0.05, label, fontdict=font, transform=ax.transAxes)

            fig.text(0.5, 0.05, labels[0], va='center', ha='center', )
            fig.text(0.07, 0.5, labels[1], va='center', ha='center', rotation='vertical')

            # create output dir if it doesnt exist
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

            fig.savefig(os.path.join(self.output_dir, plot['filename']))

    def save(self, filename):
        # save the filtered data to filename
        return
