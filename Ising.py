import numpy as np
import matplotlib.pyplot as plt
import scipy
from random import randrange, random, choice
from matplotlib import animation
from tqdm import trange
from numba.experimental import jitclass
from numba import int32, float32, types
import numba
from numba import jit, njit
import timeit

barcmap = ['CYAN', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE']
cmap1 = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']
cmap2 = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2']
cmap3 = ['crimson', 'darkmagenta', '#16a951']
cmap4 = ['crimson', '#1a5599', '#16a951']
# RED, PURPLE, GREEN, YELLOW, BLUE
cmap5_old = ['#c91f36', '#3b2e7e', '#2a6e40', '#ca6924', '#1a5599', 'darkmagenta']
cmap5 = ['#c91f36', '#3b2e7e', '#2a6e40', '#ca6924', '#1a5599', 'darkmagenta']
cmap6 = ['crimson', 'teal', '#1a5599']

spec = [
    ('N', int32),
    ('temperature', float32),
    ('mag_field', float32),
    ('mag', float32),
    ('total_ene', int32),
    ('lattice', types.Array(types.int32, 2, 'C')),
    ('nbr_table_plus', types.Array(types.int32, 1, 'C')),
    ('nbr_table_minus', types.Array(types.int32, 1, 'C')),
    ('ene_table', types.Array(types.float32, 1, 'C')),
    ('presteps', int32),
    ('sweeps', int32),
    ('ene_list', types.Array(types.float64, 1, 'C')),
    ('mag_list', types.Array(types.float64, 1, 'C')),
    ('cap_list', types.Array(types.float64, 1, 'C')),
    ('plots', types.Array(types.int32, 1, 'C')),
    ('ifpre', int32),
]


@jitclass(spec)
class Lattice2D:

    def __init__(self, N, temperature, presteps, sweeps, plots):

        self.N = N  # Size of the lattice
        self.temperature = temperature  # temperature
        self.mag_field = 0  # magnetic field

        self.mag = 0
        self.total_ene = 0

        self.lattice = np.ones((self.N, self.N), dtype=np.int32)

        self.nbr_table_plus = np.ones(self.N, dtype=np.int32)
        self.nbr_table_minus = np.ones(self.N, dtype=np.int32)
        self.ene_table = np.ones(5, dtype=np.float32)

        self.presteps = presteps
        self.sweeps = sweeps

        self.ene_list = np.ones(self.sweeps, dtype=np.float64)
        self.mag_list = np.ones(self.sweeps, dtype=np.float64)
        self.cap_list = np.ones(self.sweeps, dtype=np.float64)

        self.plots = plots
        self.ifpre = 1

        # self.nbr1D_table = np.array([0, 0, 0, 0], dtype=np.int64)

    def random_initial(self):

        for i in range(self.N):
            for j in range(self.N):
                if np.random.random() < 0.5:
                    self.lattice[i, j] = -1

    def get_nbr_table(self):

        for i in range(self.N):
            self.nbr_table_plus[i] = i + 1
            self.nbr_table_minus[i] = i - 1

        self.nbr_table_plus[self.N - 1] = 0
        self.nbr_table_minus[0] = self.N - 1

    def get_nbr(self, x, y):

        return np.array([[self.nbr_table_plus[x], y], [self.nbr_table_minus[x], y],
                         [x, self.nbr_table_plus[y]], [x, self.nbr_table_minus[y]]])

    def get_ene_table(self):
        #   0   1  2  3  4
        #   -8 -4  0  4  8
        for i in range(5):
            dE = -8 + 4 * i
            self.ene_table[i] = 1
            if dE > 0:
                self.ene_table[i] = np.exp(- dE / self.temperature)

    def get_ene_change(self, x, y):

        nbr = self.lattice[self.nbr_table_plus[x], y] + self.lattice[self.nbr_table_minus[x], y] + \
              self.lattice[x, self.nbr_table_plus[y]] + self.lattice[x, self.nbr_table_minus[y]]
        ene_change = 2 * self.lattice[x, y] * nbr
        return ene_change

    def get_ene_element(self, x, y):

        nbr = self.lattice[self.nbr_table_plus[x], y] + self.lattice[self.nbr_table_minus[x], y] + \
              self.lattice[x, self.nbr_table_plus[y]] + self.lattice[x, self.nbr_table_minus[y]]
        ene_element = - nbr * self.lattice[x, y]
        return ene_element

    def get_total_ene(self):

        self.total_ene = 0
        for x in range(self.N):
            for y in range(self.N):
                ene_element = self.get_ene_element(x, y)
                self.total_ene += ene_element / 2

    def get_capa(self):

        mean_ene_squared = (np.average(self.ene_list)) ** 2 / self.N ** 2
        mean_squared_ene = np.average(self.ene_list ** 2) / self.N ** 2

        capa = (mean_squared_ene - mean_ene_squared) / self.temperature ** 2

        return capa

    def get_susc(self):

        mean_mag_squared = (np.average(self.mag_list)) ** 2
        mean_squared_mag = np.average(self.mag_list ** 2)

        suscep = (mean_squared_mag - mean_mag_squared) / self.temperature / self.N ** 2

        return suscep

    def ising(self):

        self.get_ene_table()
        self.get_nbr_table()

        sample_index_ene = 0
        sample_index_mag = 0

        if self.ifpre == 0:
            total_steps = self.sweeps * self.N ** 2 + 1
        elif self.ifpre == 2:
            total_steps = self.presteps
        else:
            total_steps = self.presteps + self.sweeps * self.N ** 2 + 1

        for i in range(total_steps):

            x = np.random.randint(0, self.N)
            y = np.random.randint(0, self.N)

            ene_change = self.get_ene_change(x, y)
            ene_index = int(0.25 * ene_change + 2)

            if np.random.rand() < self.ene_table[ene_index]:
                self.lattice[x, y] *= -1

            if self.plots[0] or self.plots[1]:

                if i >= self.presteps:

                    if (i - self.presteps) % (self.N ** 2 * 20) == 0:

                        if self.plots[0]:
                            self.get_total_ene()
                            self.ene_list[sample_index_ene] = self.total_ene / self.N ** 2
                            sample_index_ene += 1

                        if self.plots[1]:
                            self.mag_list[sample_index_mag] = np.sum(self.lattice) / self.N**2
                            sample_index_mag += 1

    def kawasaki(self):

        for i in range(self.presteps):

            x = np.random.randint(0, self.N)
            y = np.random.randint(0, self.N)

            ene_xy = self.get_ene_element(x, y)

            nbr_ran = np.random.randint(0, 4)
            if nbr_ran == 0:
                nbr = [self.nbr_table_minus[x], y]
            if nbr_ran == 1:
                nbr = [self.nbr_table_plus[x], y]
            if nbr_ran == 2:
                nbr = [x, self.nbr_table_minus[y]]
            else:
                nbr = [x, self.nbr_table_plus[y]]

            ene_nbr = self.get_ene_element(nbr[0], nbr[1])

            ene_before = ene_xy + ene_nbr

            temp = self.lattice[x, y]
            self.lattice[x, y] = self.lattice[nbr[0], nbr[1]]
            self.lattice[nbr[0], nbr[1]] = temp

            ene_xy = self.get_ene_element(x, y)
            ene_nbr = self.get_ene_element(nbr[0], nbr[1])

            ene_after = ene_xy + ene_nbr

            ene_diff = ene_after - ene_before

            if np.random.rand() > np.exp(-ene_diff / self.temperature):
                temp = self.lattice[x, y]
                self.lattice[x, y] = self.lattice[nbr[0], nbr[1]]
                self.lattice[nbr[0], nbr[1]] = temp

class Plots:

    def __init__(self, temperature, size, presteps, sweeps, initial, ):

        self.temperature = temperature
        self.size = size

        self.presteps = presteps
        self.sweeps = sweeps

        self.plots_option = np.array([0, 0], dtype=np.int32)

        self.initial = initial

        self.labelFontsize = 12
        self.titleFontsize = 12

    def plot_ising(self, multiPlots=None):

        print('Plotting ising')

        lattice = Lattice2D(self.size, self.temperature, self.presteps, self.sweeps, self.plots_option)
        lattice.ifpre = 2
        # lattice.ising()
        lattice.ising()

        if self.initial == 'random':
            lattice.random_initial()

        tbar = trange(multiPlots[0])

        if isinstance(multiPlots, list):
            print('Start plotting')
            iterations = 0
            ising_fig, ising_ax = plt.subplots(multiPlots[0], multiPlots[1], figsize=(10, 10))
            lattice.ifpre = 0
            for i in tbar:
                for j in range(multiPlots[1]):
                    plt.xticks([])
                    plt.yticks([])
                    ising_ax[i, j].imshow(lattice.lattice, cmap='gray')
                    # ising_ax[i, j].set_xlabel('{:.0e} runs'.format((self.sweeps + self.presteps) * iterations),
                    #                           fontsize=10)
                    lattice.ising()

                    iterations += 1

            plt.tight_layout()
            plt.show()

            # self.plot_mag_runs()

        else:
            lattice.ising()
            plt.imshow(lattice.lattice, cmap='gray')
            plt.xlabel('{} runs'.format(self.steps))
            plt.title('Metropolis simulationg of Ising Model\n Initial state is {0}, '
                      '$T$={1:.1f}, $N$={2:d}'.format(self.initial, self.temperature, self.size),
                      fontsize=20)
            plt.tight_layout()
            plt.show()

    def plot_kawasaki(self, multiPlots):

        lattice = Lattice2D(self.size, self.temperature, self.presteps, self.sweeps, self.plots_option)

        lattice.random_initial()
        print(lattice.lattice)

        tbar = trange(multiPlots[0])

        if isinstance(multiPlots, list):
            print('Start plotting')
            ising_fig, ising_ax = plt.subplots(multiPlots[0], multiPlots[1], figsize=(10, 10))
            for i in tbar:
                for j in range(multiPlots[1]):
                    ising_ax[i, j].imshow(lattice.lattice, cmap='gray')
                    lattice.kawasaki()

            plt.tight_layout()
            plt.show()


    def plot_mag(self, temp_list):

        self.plots_option = np.array([0, 1], dtype=np.int32)
        mag_list = np.array([])
        tbar = trange(len(temp_list))
        for i in tbar:
            lattice = Lattice2D(self.size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
            if self.initial == 'random':
                lattice.random_initial()
            lattice.ising()
            mag_list = np.append(mag_list, np.average(lattice.mag_list))

        plt.scatter(temp_list, mag_list, color='crimson', marker='o', alpha=0.6, s=7, )
        plt.plot(temp_list, mag_list, color='crimson', lw=1.6)
        plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        plt.ylabel(r'$\langle M \rangle$', fontsize=self.labelFontsize)
        # plt.grid()
        # plt.show()
        plt.savefig('mag_of_size{}.pdf'.format(self.size))

        return mag_list

    def plot_mag_size(self, temp_list, size_list):

        self.plots_option = np.array([0, 1], dtype=np.int32)

        mag_size_fig, mag_size_ax = plt.subplots(figsize=(4, 4))

        sbar = trange(len(size_list))
        tbar = trange(len(temp_list))
        for i in sbar:
            mag_list = np.array([])
            for j in tbar:
                lattice = Lattice2D(size_list[i], temp_list[j], self.presteps, self.sweeps, self.plots_option)
                lattice.ising()
                mag_list = np.append(mag_list, np.average(lattice.mag_list))

            mag_size_ax.scatter(temp_list, mag_list, s=7, alpha=0.6, marker='o', color=cmap6[i])
            mag_size_ax.plot(temp_list, mag_list, lw=1.6, color=cmap6[i],
                             label='{}'.format(size_list[i]))

        plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        plt.ylabel(r'$\langle M \rangle$', fontsize=self.labelFontsize)

        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig('figure 3', format='png', dpi=1000)

    def plot_ene(self, temp_list):

        ene_list = np.array([])

        self.plots_option = np.array([1, 0], dtype=np.int32)

        tbar = trange(len(temp_list))
        for i in tbar:
            ene_list_temp = np.array([])
            lattice = Lattice2D(self.size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
            lattice.ising()
            ene_list_dir = lattice.ene_list
            for i in range(len(ene_list_dir)):
                if ene_list_dir[i] != 1:
                    ene_list_temp = np.append(ene_list_temp, ene_list_dir[i])

            ene_list = np.append(ene_list, np.average(ene_list_temp))
            # ene_list = np.append(ene_list, np.average(lattice.ene_list))

        plt.plot(temp_list, ene_list, color='crimson')
        plt.scatter(temp_list, ene_list, color='crimson', s=10)
        plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        plt.ylabel(r'$\langle E\rangle$', fontsize=self.labelFontsize)
        plt.show()
        return ene_list

    def plot_ene_size(self, temp_list, size_list):

        self.plots_option = np.array([1, 0], dtype=np.int32)

        ene_list = np.array([])
        ene_fig, ene_ax = plt.subplots(figsize=(4, 4))

        tbar = trange(len(temp_list))
        sbar = trange(len(size_list))
        for i in sbar:
            for j in tbar:
                lattice = Lattice2D(self.size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
                lattice.ising()
                ene_list = np.append(ene_list, lattice.get_mean_ene()[0])

            ene_ax.scatter(temp_list, ene_list, color='crimson', s=10)
            ene_ax.xlabel(r'$T$', fontsize=self.labelFontsize)
            plt.ylabel(r'$\langle E\rangle$', fontsize=self.labelFontsize)
        plt.show()

    def plot_cumulant(self, temp_list):

        self.plots_option = np.array([0, 1], dtype=np.int32)
        mag_list = np.array([])
        mag_sq_list = np.array([])
        mag_quar_list = np.array([])
        cumu_list = np.array([])
        tbar = trange(len(temp_list))
        for i in tbar:
            lattice = Lattice2D(self.size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
            if self.initial == 'random':
                lattice.random_initial()
            lattice.ising()
            mag_list = np.append(mag_list, np.average(lattice.mag_list))
            mag_sq_list = np.append(mag_sq_list, np.average(np.square(lattice.mag_list)))
            mag_quar_list = np.append(mag_quar_list, np.average(np.square(np.square(lattice.mag_list))))

        for j in range(len(mag_list)):
            cumulant = 1 - mag_quar_list[j] / (3 * mag_sq_list[j] ** 2)
            cumu_list = np.append(cumu_list, cumulant)

        cumu_fig, cumu_ax = plt.subplots()
        cumu_ax.plot(temp_list, cumu_list, color=cmap5[1], lw=1.6, )

        # mag_mom_fig, mag_mom_ax = plt.subplots()
        # mag_mom_ax.plot(temp_list, mag_list, color=cmap5[0], lw=1.6, label=r'$\langle m \rangle$')
        # mag_mom_ax.plot(temp_list, mag_sq_list, color=cmap5[1], lw=1.6, label=r'$\langle m^2 \rangle$')
        # mag_mom_ax.plot(temp_list, mag_quar_list, color=cmap5[2], lw=1.6, label=r'$\langle m^4 \rangle$')

        # mag_mom_ax.scatter(temp_list, mag_list,      color=cmap5[0], marker='o', s=8, alpha=0.6)
        # mag_mom_ax.scatter(temp_list, mag_sq_list,   color=cmap5[1], marker='o', s=8, alpha=0.6)
        # mag_mom_ax.scatter(temp_list, mag_quar_list, color=cmap5[2], marker='o', s=8, alpha=0.6)

        # plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        # plt.ylabel('Magnetization Moments', fontsize=self.labelFontsize)
        # plt.grid()
        # plt.legend()
        plt.tight_layout()
        plt.show()

    def cumu_critical(self, temp_list, size):

        self.plots_option = np.array([0, 1], dtype=np.int32)
        mag_list = np.array([])
        mag_sq_list = np.array([])
        mag_quar_list = np.array([])
        cumu_list = np.array([])
        tbar = trange(len(temp_list))

        for i in tbar:
            lattice = Lattice2D(size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
            lattice.ising()
            mag_list = np.append(mag_list, np.average(lattice.mag_list))
            mag_sq_list = np.append(mag_sq_list, np.average(np.square(lattice.mag_list)))
            mag_quar_list = np.append(mag_quar_list, np.average(np.square(np.square(lattice.mag_list))))

        for j in range(len(mag_list)):
            cumulant = 1 - mag_quar_list[j] / (3 * mag_sq_list[j] ** 2)
            cumu_list = np.append(cumu_list, cumulant)

        return cumu_list

    def plot_cumu_critical(self, temp_list):

        # cumu_list_50 = self.cumu_critical(temp_list, 50)
        # cumu_list_5 = self.cumu_critical(temp_list, 5)
        cumu_list_8 = self.cumu_critical(temp_list, 64)
        # cumu_list_20 = self.cumu_critical(temp_list, 20)
        # cumu_list_15 = self.cumu_critical(temp_list, 15)
        # cumu_list_80 = self.cumu_critical(temp_list, 80)

        # cumu_crit_list_1 = np.zeros(len(cumu_list_50))
        # cumu_crit_list_2 = np.zeros(len(cumu_list_50))
        # cumu_crit_list_3 = np.zeros(len(cumu_list_50))

        # for i in range(len(cumu_list_50)):
        #     cumu_crit_list_1[i] = cumu_list_50[i] / cumu_list_5[i]
        #
        # for i in range(len(cumu_list_50)):
        #     cumu_crit_list_2[i] = cumu_list_8[i] / cumu_list_20[i]

        # for i in range(len(cumu_list_50)):
        #     cumu_crit_list_3[i] = cumu_list_15[i] / cumu_list_80[i]

        fig, ax = plt.subplots()
        ax.plot(temp_list, cumu_list_8)
        # ax.plot(temp_list, cumu_crit_list_1, color='crimson', label='50/5')
        # ax.plot(temp_list, cumu_crit_list_2, color='teal', label='8/20')
        # ax.plot(temp_list, cumu_crit_list_3, color='#1a5599', label='15/80')
        # plt.xlabel(r'$T$')
        # plt.ylabel(r'$U_L/U'_L$')
        # plt.legend()
        plt.show()
        #
        return cumu_list_8

    def plot_mag_runs(self, bins=50):

        self.plots_option = np.array([0, 1], dtype=np.int32)

        lattice = Lattice2D(self.size, self.temperature, self.presteps, self.sweeps, self.plots_option)

        if self.initial == 'random':
            lattice.random_initial()

        lattice.ising()
        # print(lattice.mag_list[::200])
        # plt.plot(lattice.mag_list, lw=1.6, color='k', alpha=1)

        plt.hist(lattice.mag_list, bins, facecolor='g', alpha=0.75)
        plt.tight_layout()
        plt.show()
        return lattice.mag_list

    def plot_ising_mag(self, multiPlots):

        plots_number = multiPlots[0] * multiPlots[1]

        mag_list = np.array([])

        self.plots_option = np.array([0, 1], dtype=np.int32)

        lattice = Lattice2D(self.size, self.temperature, 0, self.steps / plots_number, self.plots_option)

        if self.initial == 'random':
            lattice.random_initial()

        tbar = trange(multiPlots[0])

        iterations = 0
        ising_fig, ising_ax = plt.subplots(multiPlots[0], multiPlots[1], figsize=(10, 4))
        mag_fig, mag_ax = plt.subplots(figsize=(10, 1.5))
        for i in tbar:
            for j in range(multiPlots[1]):
                ising_ax[i, j].imshow(lattice.lattice, cmap='gray')
                ising_ax[i, j].set_xlabel('{:.0e} runs'.format((self.steps + self.presteps) * iterations),
                                          fontsize=10)

                lattice.ising()
                mag_list = np.append(mag_list, lattice.mag_list)

                iterations += 1

        ising_fig.suptitle('Metropolis simulationg of Ising Model\n Initial state is {0}, '
                           '$T$={1:.1f}, $Size$={2:d}'.format(self.initial, self.temperature, self.size),
                           fontsize=20)
        ising_fig.tight_layout()
        plt.show()
        plt.savefig('animate', format='png', dpi=1000)

        mag_ax.plot(mag_list, lw=0.8, color='k')
        mag_fig.tight_layout()
        plt.show()
        plt.savefig('magis', format='png', dpi=1000)

    def plot_capa(self, temp_list):

        capa_list = np.array([])
        self.plots_option = np.array([1, 0], dtype=np.int32)

        tbar = trange(len(temp_list))
        for i in tbar:
            lattice = Lattice2D(self.size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
            lattice.ising()
            print(lattice.ene_list)
            capa_list = np.append(capa_list, lattice.get_capa())

        plt.plot(temp_list, capa_list, color='crimson', lw=1, ls='--')
        plt.scatter(temp_list, capa_list, color='crimson', s=10)
        plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        plt.ylabel(r'$C_V$', fontsize=self.labelFontsize)
        plt.show()

        return capa_list

    def plot_susc(self, temp_list):

        susc_list = np.array([])
        self.plots_option = np.array([0, 1], dtype=np.int32)

        tbar = trange(len(temp_list))
        for i in tbar:
            lattice = Lattice2D(self.size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
            lattice.ising()
            susc_list = np.append(susc_list, lattice.get_susc())

        plt.plot(temp_list, susc_list, color='crimson', lw=1, ls='--')
        plt.scatter(temp_list, susc_list, color='crimson', s=10)
        plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        plt.ylabel(r'$\chi$', fontsize=self.labelFontsize)
        # plt.grid()
        plt.show()

        return susc_list

    def plot_capa_size(self, temp_list, size_list):

        self.plots_option = np.array([1, 0], dtype=np.int32)

        sbar = trange(len(size_list))
        tbar = trange(len(temp_list))
        for i in sbar:
            capa_list = np.array([])
            for j in tbar:
                lattice = Lattice2D(size_list[i], temp_list[j], self.presteps, self.sweeps, self.plots_option)
                lattice.ising()
                capa_list = np.append(capa_list, lattice.get_capa())

            # capa_size_ax.scatter(temp_list, capa_list, s=10, alpha=0.6, color=cmap5[i])
            plt.gca().plot(temp_list, capa_list, lw=1.6, color='gray', alpha=0.5,
                           label='{}'.format(size_list[i]))

        plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        plt.ylabel(r'$C_V$', fontsize=self.labelFontsize)
        # plt.grid()
        plt.show()
        plt.legend()
        # plt.savefig('capa', format='png', dpi=1000)

    def plot_capa_sweep(self, temp_list):

        self.plots_option = np.array([1, 0], dtype=np.int32)

        capcity_size_fig, capa_size_ax = plt.subplots()

        sweep_list = [1, 2, 3, 4, 5, 6, 7]
        tbar = trange(len(temp_list))
        sbar = trange(len(sweep_list))

        for j in sbar:
            capa_list = np.array([])
            for i in tbar:
                lattice = Lattice2D(self.size, temp_list[i], self.presteps * sweep_list[j], self.sweeps,
                                    self.plots_option)
                lattice.ising()
                capa_list = np.append(capa_list, lattice.get_capa())

            plt.gca().plot(temp_list, capa_list, lw=1.6, color='gray', alpha=0.5, )

        # plt.grid()
        plt.show()
        plt.legend()
        # plt.savefig('capa', format='png', dpi=1000)

    def plot_ene_to_size(self, size_list):

        ene_list = np.array([])

        tbar = trange(len(size_list))
        for i in tbar:
            lattice = Lattice2D(size_list[i], self.temperature)
            lattice.ising()
            ene_list = np.append(ene_list, lattice.get_mean_ene()[0])

        plt.scatter(size_list, ene_list, color='crimson', s=10)
        plt.xlabel('size', fontsize=self.labelFontsize)
        plt.ylabel('ene', fontsize=self.labelFontsize)
        plt.title('ene at varied size',
                  fontsize=self.titleFontsize)
        plt.grid()
        plt.show()

    def plot_memlost(self, temp_list):

        self.plots_option = np.array([0, 1], dtype=np.int32)

        mem_fig, mem_ax = plt.subplots()

        tbar = trange(len(temp_list))

        num_of_tests = 1

        for i in range(num_of_tests):
            for j in tbar:
                lattice = Lattice2D(self.size, temp_list[j], 0, self.sweeps, self.plots_option)
                lattice.random_initial()
                lattice.ising()
                mem_ax.scatter(np.arange(0, self.sweeps, 500), lattice.mag_list[::500], s=4, color=cmap1[i])
                # mem_ax.scatter(np.arange(0, self.sweeps, 10), lattice.mag_list[::10], s=5, color=cmap5[i])

        # plt.xlabel('$T$', fontsize=self.labelFontsize)
        # plt.ylabel('$\langle M \rangle$', fontsize=self.labelFontsize)
        plt.tight_layout()
        plt.show()

    def plot_capa_final(self, ):

        self.plots_option = np.array([1, 0], dtype=np.int32)

        temp_list = np.arange(1, 4, 0.1)
        size_list = np.arange(8, 64, 2)
        peak_list = np.empty(2)
        capa_list = np.array([])
        sbar = trange(len(size_list))
        tbar = trange(len(temp_list))
        for i in sbar:
            capa_list = np.array([])
            for j in tbar:
                lattice = Lattice2D(size_list[i], temp_list[j], self.presteps, self.sweeps, self.plots_option)
                lattice.ising()
                capa_list = np.append(capa_list, lattice.get_capa())
            # print(temp_list[np.argmax(capa_list)], np.max(capa_list))
            peak_list = np.vstack((peak_list,
                                   np.array([temp_list[np.argmax(capa_list)], np.max(capa_list)])
                                   ))

            if size_list[i] in [8, 16, 32, 64]:
                plt.gca().plot(temp_list, capa_list, lw=1.6, color='#1a5599', alpha=0.5, )
            else:
                plt.gca().plot(temp_list, capa_list, lw=1, color='gray', alpha=0.5, )

        peak_list = np.delete(peak_list, 0, axis=0)

        plt.gca().plot(peak_list[:, 0], peak_list[:, 1], lw=1.2, linestyle='dashed', color='teal')
        #
        # plt.xlabel(r'$T$', fontsize=self.labelFontsize)
        # plt.ylabel(r'$C_V$', fontsize=self.labelFontsize)
        # # plt.grid()
        # plt.show()
        # plt.savefig('capa', format='png', dpi=1000)

    def plot_mag_cluster(self, temp_list):

        # self.plots_option = np.array([0, 1], dtype=np.int32)
        mag_list = np.array([])
        tbar = trange(len(temp_list))
        for i in tbar:
            lattice = Lattice2D(self.size, temp_list[i], self.presteps, self.sweeps, self.plots_option)
            lattice.nbr1D()
            if self.initial == 'random':
                lattice.random_initial()
            mag_list_clus = lattice.cluster()
            mag_list = np.append(mag_list, np.average(mag_list_clus))

        plt.scatter(temp_list, mag_list, color='crimson', marker='o', alpha=0.6, s=18, )
        plt.plot(temp_list, mag_list, color='crimson', lw=1.6)
        # plt.grid()
        plt.show()


PRESTEPS = 10000 * 64**2
SWEEPS = 50000 * 20
SIZE = 64
TEMP = 1.5
plots_option = np.array([0, 0], dtype=np.int32)
temp_list = np.arange(1, 4, 0.01)

Plots = Plots(20, size=SIZE, presteps=PRESTEPS, sweeps=SWEEPS, initial='normal')


