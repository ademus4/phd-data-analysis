import os
import matplotlib.pylab as plt
import numpy as np
import uproot3
import uproot4
from concurrent.futures import ThreadPoolExecutor


class Analysis:
    def __init__(self, output_dir, n_workers=4, cache="4 GB"):
        self.output_dir = output_dir
        os.mkdir(self.output_dir)

        self.cut_mm2=0.01
        self.cut_mE=0.5
        self.cut_mP=0.5
        self.cut_mm2pi=(0.8, 1.1)
        self.tree_data = []
        self.tree_data_cut = []

        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        self.cache = uproot4.LRUArrayCache(cache)

        plt.style.use('seaborn')
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

    def load_data(self, path):
        print("Loading:")
        print(path)
        self.tree_data = uproot4.lazy(path, 
                                      executor=self.executor, 
                                      blocking=False, 
                                      cache=self.cache)
        #self.apply_cuts()

    def apply_cuts(self):
        cuts = (
            #np.array(self.tree_data['Pi2TriggerMesonex']==1) & 
            np.array(np.abs(self.tree_data['Pi2MissMass2'])<self.cut_mm2) &
            np.array(np.abs(self.tree_data['Pi2MissE'])<self.cut_mE) &
            np.array(self.tree_data['Pi2MissP']<self.cut_mE) &
            np.array(
                (self.cut_mm2pi[0]<self.tree_data['Pi2MissMassnP']) & 
                (self.tree_data['Pi2MissMassnP']<self.cut_mm2pi[1]))
        )
        self.tree_data_cut = self.tree_data[cuts]

    def plot_exc_cuts(self):
        params = {
            'bins': 201,
            'histtype': 'step',
            'linewidth': 2
        }

        fig, axes = plt.subplots(2, 2, figsize=(16,12))
        axes = axes.flatten()
        h = axes[0].hist(np.array(self.tree_data['Pi2MissMass2']), range=(-1, 1), **params)
        axes[0].axvline(-self.cut_mm2, color='red')
        axes[0].axvline(self.cut_mm2, color='red')
        axes[0].set_ylabel('Events')
        axes[0].set_xlabel('Missing Mass ^2')

        h = axes[1].hist(np.array(self.tree_data['Pi2MissE']), range=(-2, 2), **params)
        axes[1].axvline(-self.cut_mE, color='red')
        axes[1].axvline(self.cut_mE, color='red')
        axes[1].set_xlabel('Missing Energy')

        h = axes[2].hist(np.array(self.tree_data['Pi2MissP']), range=(-1, 1), **params)
        axes[2].axvline(self.cut_mP, color='red')
        axes[2].set_xlabel('Missing Momentum')


        h = axes[3].hist(np.array(self.tree_data['Pi2MissMassnP']), range=(0, 2), **params)
        axes[3].set_xlabel('Missing Mass 2pi')
        axes[3].axvline(self.cut_mm2pi[0], color='red')
        axes[3].axvline(self.cut_mm2pi[1], color='red')
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'exclusivity_cuts.png'))

    def plot_timing(self, xrange=(-0.5, 0.5)):
        vals = [
            "ElectronDeltaTime",
            "ProtonDeltaTime",
            "PimDeltaTime",
            "PipDeltaTime"
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
        fig.savefig(os.path.join(self.output_dir, 'timing_plots.png'))

    def plot_mesons(self, cuts=False):
        params = {
            'bins': 201,
            'histtype': 'step',
            'linewidth': 2
        }

        if cuts:
            data = self.tree_data_cut
        else:
            data = self.tree_data

        vals = [
            ["Pi2MesonMass", (0, 3), 'Mass [GeV/$c^2$]'],
            ["Pi2DppMass",   (1, 4), 'Mass [GeV/$c^2$]'],
            ["Pi2D0Mass",    (1, 4), 'Mass [GeV/$c^2$]'],
            ["Pi2t",         (0, 5), '-t [GeV/c]'],
        ]

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        axes = ax.flatten()
        for i, valr in enumerate(vals):
            val, r , xlab = valr
            h = axes[i].hist(np.abs(np.array(data[val])), range=r, **params)
            axes[i].set_title(val)
            axes[i].set_ylabel('Events (per bin)')
            axes[i].set_xlabel(xlab)

        filename = 'meson_plots.png'
        if cuts:
            filename = 'cut_' + filename
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename))


    def plot_electron(self, cuts=False):
        params = {
            'histtype': 'step',
            'linewidth': 2
        }

        if cuts:
            data = self.tree_data_cut
        else:
            data = self.tree_data

        vals = [
            ["ElectronP",      (0, 7), 201, 'Momentum [GeV/c]'],
            ["ElectronTheta",  (1, 7), 201, '$\\theta$ [deg]'],
            ["ElectronPhi",    (-180, 180), 201, '$\phi$ [deg]'],
            ["ElectronDeltaE", (0, 30), 201, 'Energy [MeV?]'],
            ["ElectronRegion", (0, 5000), 21, 'Region'],
            ["ElectronSector", (0, 8), 8, 'Sector']
        ]

        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        axes = ax.flatten()
        for i, valr in enumerate(vals):
            val, r, bins, xlab = valr
            h = axes[i].hist(np.array(data[val]), range=r, bins=bins, **params)
            axes[i].set_title(val)
            axes[i].set_ylabel('Events (per bin)')
            axes[i].set_xlabel(xlab)

        filename = 'electron_plots.png'
        if cuts:
            filename = 'cut_' + filename
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename))


    def plot_proton(self, cuts=False):
        params = {
            'histtype': 'step',
            'linewidth': 2
        }

        regions = [1000, 2000, 3000]

        if cuts:
            data = self.tree_data_cut
        else:
            data = self.tree_data

        vals = [
            ["ProtonP",      (0, 7), 201, 'Momentum [GeV/c]'],
            ["ProtonTheta",  (0, 110), 201, '$\\theta$ [deg]'],
            ["ProtonPhi",    (-180, 180), 201, '$\phi$ [deg]'],
            ["ProtonDeltaE", (0, 70), 201, 'Energy [MeV?]'],
            ["ProtonRegion", (0, 5000), 21, 'Region'],
            ["ProtonSector", (0, 8), 8, 'Sector']
        ]

        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        axes = ax.flatten()
        for i, valr in enumerate(vals):
            val, r, bins, xlab = valr
            for region in regions:
                h = axes[i].hist(np.array(data[data['ProtonRegion']==region][val]), range=r, bins=bins, label=f"{region}", **params)
            axes[i].set_title(val)
            axes[i].set_ylabel('Events (per bin)')
            axes[i].set_xlabel(xlab)

        filename = 'proton_plots.png'
        if cuts:
            filename = 'cut_' + filename
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename))


    def plot_pip(self, cuts=False):
        params = {
            'histtype': 'step',
            'linewidth': 2
        }

        regions = [1000, 2000, 3000]

        if cuts:
            data = self.tree_data_cut
        else:
            data = self.tree_data

        vals = [
            ["PipP",      (0, 7), 201, 'Momentum [GeV/c]'],
            ["PipTheta",  (0, 110), 201, '$\\theta$ [deg]'],
            ["PipPhi",    (-180, 180), 201, '$\phi$ [deg]'],
            ["PipDeltaE", (0, 70), 201, 'Energy [MeV?]'],
            ["PipRegion", (0, 5000), 21, 'Region'],
            ["PipSector", (0, 8), 8, 'Sector']
        ]

        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        axes = ax.flatten()
        for i, valr in enumerate(vals):
            val, r, bins, xlab = valr
            for region in regions:
                h = axes[i].hist(np.array(data[data['PipRegion']==region][val]), range=r, bins=bins, label=f"{region}", **params)
            axes[i].set_title(val)
            axes[i].set_ylabel('Events (per bin)')
            axes[i].set_xlabel(xlab)
            axes[i].legend()

        filename = 'pip_plots.png'
        if cuts:
            filename = 'cut_' + filename
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename))


    def plot_pim(self, cuts=False):
        params = {
            'histtype': 'step',
            'linewidth': 2
        }

        regions = [1000, 2000, 3000]

        if cuts:
            data = self.tree_data_cut
        else:
            data = self.tree_data

        vals = [
            ["PimP",      (0, 7), 201, 'Momentum [GeV/c]'],
            ["PimTheta",  (0, 80), 201, '$\\theta$ [deg]'],
            ["PimPhi",    (-180, 180), 201, '$\phi$ [deg]'],
            ["PimDeltaE", (0, 40), 201, 'Energy [MeV?]'],
            ["PimRegion", (0, 5000), 21, 'Region'],
            ["PimSector", (0, 7), 7, 'Sector']
        ]

        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        axes = ax.flatten()
        for i, valr in enumerate(vals):
            val, r, bins, xlab = valr
            for region in regions:
                h = axes[i].hist(np.array(data[data['PimRegion']==region][val]), range=r, bins=bins, label=f"{region}", **params)
            axes[i].set_title(val)
            axes[i].set_ylabel('Events (per bin)')
            axes[i].set_xlabel(xlab)
            axes[i].legend()

        filename = 'pim_plots.png'
        if cuts:
            filename = 'cut_' + filename
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename))


    def plot_meson_2D(self, cuts=False):
        val1 = "Pi2MesonMass"
        val2 = "Pi2D0Mass"
        val3 = "Pi2DppMass"

        if cuts:
            data = self.tree_data_cut
        else:
            data = self.tree_data

        params = {
            'range': [[0.5, 2.5], [1, 3.5]],
            'bins': 100,
        }

        fig, axes = plt.subplots(1, 2, figsize=(16,6))
        axes[0].hist2d(
            np.array(data[val1]), 
            np.array(data[val2]),
            **params)
        axes[0].set_xlabel(val1)
        axes[0].set_ylabel(val2)

        axes[1].hist2d(
            np.array(data[val1]), 
            np.array(data[val3]),
            **params)
        axes[1].set_xlabel(val1)
        axes[1].set_ylabel(val3)
        
        filename = 'meson_2D_plots.png'
        if cuts:
            filename = 'cut_' + filename
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename))

    def plot_meson_decay_angle(self, cuts=False):
        if cuts:
            data = self.tree_data_cut
        else:
            data = self.tree_data

        vals = ['Pi2MesonMass', 'Pi2MesonGJPhi']

        # mass bins for meson
        mbwidth = 0.2
        mbins = np.arange(0, 2, mbwidth)

        params = {
            'range': [[0, 3.5], [-3.5, 3.5]],
            'bins': 100,
        }

        font = {
            'family': 'serif',
            'color':  'white',
            'weight': 'normal',
            'size': 12,
        }

        fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
        for i, ax in enumerate(axes.flatten()):
            if i>=len(mbins):
                ax.remove()
                continue
            mass_low = mbins[i]
            mass_high = mass_low + mbwidth
            data = data[np.array(-data['Pi2t']>=mass_low) & np.array(-data['Pi2t']<=mass_high)]
            x, y = [np.array(data[vals[0]]), np.array(data[vals[1]])]
            ax.hist2d(x, y, **params)
            ax.text(2.1, -3.2, '{:.2f}>-t>{:.2f}'.format(mass_low, mass_high), fontdict=font)

        fig.text(0.5, 0.05, "Meson Mass [GeV/$c^2$]", va='center', ha='center', )
        fig.text(0.07, 0.5, "Meson Decay Angle (GJ) Phi", va='center', ha='center', rotation='vertical')

        filename = 'meson_decay_angle_plots.png'
        if cuts:
            filename = 'cut_' + filename
        fig.savefig(os.path.join(self.output_dir, filename))
        

    def save(self, filename):
        # save the filtered data to filename
        return
