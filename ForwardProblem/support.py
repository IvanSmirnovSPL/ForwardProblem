from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


class Residual:
    debit: list = []
    pressure: list = []
    h: list = []

    def show(self):
        fig, ax = plt.subplots(3, 1)
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]

        def f(x, a, b):
            return a * x + b

        popt = curve_fit(f, np.log(np.array(self.h)), np.log(np.array(self.debit)))[0]
        ax1.scatter(np.log(np.array(self.h)), np.log(np.array(self.debit)), color='b',
                    label='debit')
        ax1.plot(np.log(np.array(self.h)), f(np.log(np.array(self.h)), *popt), color='r',
                 label=f' $y = {"%.1f" % popt[0]} \cdot x {"+" if popt[1] > 0 else "-"} {"%.1e" % abs(popt[1])}$')
        ax1.grid()
        ax1.legend()
        popt = curve_fit(f, np.log(np.array(self.h)), np.log(np.array(self.pressure)))[0]
        ax2.scatter(np.log(np.array(self.h)), np.log(np.array(self.pressure)), color='b',
                    label='pressure')
        ax2.plot(np.log(np.array(self.h)), f(np.log(np.array(self.h)), *popt), color='r',
                 label=f' $y = {"%.1f" % popt[0]} \cdot x {"+" if popt[1] > 0 else "-"} {"%.1e" % abs(popt[1])}$')
        ax2.grid()
        ax2.legend()
        self.debit = np.array(self.debit) / max(self.debit)
        self.pressure = np.array(self.pressure) / max(self.pressure)
        popt = curve_fit(f, np.log(np.array(self.h)), np.log(np.array(self.debit) + np.array(self.pressure)))[0]
        ax3.scatter(np.log(np.array(self.h)), np.log(np.array(self.debit) + np.array(self.pressure)), color='b',
                    label='debit + pressure')
        ax3.plot(np.log(np.array(self.h)), f(np.log(np.array(self.h)), *popt), color='r',
                 label=f' $y = {"%.1f" % popt[0]} \cdot x {"+" if popt[1] > 0 else "-"} {"%.1e" % abs(popt[1])}$')
        ax3.grid()
        ax3.legend()
        plt.savefig(r'images\residual.png', dpi=500)
