import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    app = mo.App()
    return app, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Calculating convolutions

    This activity is designed to support Lecture 2 of the Advanced Molecular Spectroscopy course.

    The demos below illustrate the calculation of the convolution (black trace, lower panel) of two distributions.  Move the sliders to change the width of the two distributions and their relative offset τ. At each value of τ, the shaded yellow area shows the integral of the overlap of the two distributions, which is the value of the convolution at that τ point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This first demo shows the convolution of a gaussian and a lorentzian distribution. This is the signal you would measure if you tried to probe a transition with a natural (Lorentzian) lineshape (e.g. the excited state of a fluorescent gas-phase molecule that returns to the ground state via spontaneous emission) using a gaussian laser pulse.  Experiment with moving the sliders and note what happens to the shape and width of the measured signal when the gaussian is much wider or much narrower than the lorentzian. When does the measured signal truly represent the properties of the molecule being probed? How wide can you make the guassian (laser pulse) before the measured signal no longer properly represents the system being measured?
    """)
    return


@app.cell
def _(mo):
    gamma = mo.ui.slider(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=0.5,
        label="Lorentzian width γ",
    )

    sigma = mo.ui.slider(
        start=0.1,
        stop=5.0,
        step=0.1,
        value=2.0,
        label="Gaussian width σ",
    )

    tau = mo.ui.slider(
        start=-10.0,
        stop=10.0,
        step=0.1,
        value=0.0,
        label="Gaussian centre τ",
    )
    return gamma, sigma, tau


@app.cell
def _(gamma, mo, sigma, tau):
    mo.vstack([
        gamma,
        sigma,
        tau,
    ])
    return


@app.cell
def _():
    LOR_COLOUR = (187/255, 85/255, 102/255)
    GAUSS_COLOUR = (0/255, 68/255, 136/255)
    OVERLAP_COLOUR = (221/255, 170/255, 51/255)
    return GAUSS_COLOUR, LOR_COLOUR, OVERLAP_COLOUR


@app.cell
def _(GAUSS_COLOUR, LOR_COLOUR, OVERLAP_COLOUR, gamma, np, plt, sigma, tau):
    def lorentzian(t, gamma):
        return (gamma / np.pi) / (t**2 + gamma**2)

    def gaussian(t, sigma, gamma):
        peak = 1.0 / (np.pi * gamma)

        return peak * np.exp(
            -(t**2) / (2 * sigma**2)
        )

    # --------------------------------------------------
    # Top-panel data
    # --------------------------------------------------

    t = np.linspace(-15, 15, 4000)

    L = lorentzian(t, gamma.value)

    G = gaussian(
        t - tau.value,
        sigma.value,
        gamma.value,
    )

    overlap = np.minimum(L, G)

    current_conv = np.trapezoid(L * G, t)

    # --------------------------------------------------
    # Precompute convolution trace
    # --------------------------------------------------

    tau_grid = np.linspace(-10, 10, 300)

    conv_values = []

    for tau_i in tau_grid:

        G_i = gaussian(
            t - tau_i,
            sigma.value,
            gamma.value,
        )

        conv_values.append(
            np.trapezoid(L * G_i, t)
        )

    conv_values = np.array(conv_values)

    # Find which part of the trace should be drawn
    mask = tau_grid <= tau.value

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # --------------------------------------------------
    # Top panel
    # --------------------------------------------------

    ax1.plot(
        t,
        L,
        color=LOR_COLOUR,
        lw=3,
        label="Lorentzian",
    )

    ax1.plot(
        t,
        G,
        color=GAUSS_COLOUR,
        lw=3,
        label="Gaussian",
    )

    ax1.fill_between(
        t,
        0,
        overlap,
        color=OVERLAP_COLOUR,
        alpha=0.8,
    )

    ax1.set_xlim(-15, 15)

    ax1.set_ylabel("Amplitude")

    ax1.set_title(
        f"Current convolution value = {current_conv:.4f}"
    )

    ax1.legend(frameon=False)

    # --------------------------------------------------
    # Bottom panel
    # --------------------------------------------------

    ax2.plot(
        tau_grid[mask],
        conv_values[mask],
        color="black",
        lw=2.5,
    )

    ax2.plot(
        tau.value,
        current_conv,
        "o",
        color="black",
        markersize=7,
    )

    ax2.set_xlim(
        -15,
        15,
    )

    ax2.set_ylim(
        0,
        1.05 * conv_values.max(),
    )

    ax2.axvline(
        tau.value,
        color="grey",
        ls="--",
        alpha=0.5,
    )

    ax2.set_xlabel("Gaussian centre τ")
    ax2.set_ylabel("Convolution")

    fig.tight_layout()

    fig
    return gaussian, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This second demo shows the convolution of two gaussian distributions. This is the signal you would measure if you used a guassian laser pulse to probe a transition that also had a gaussian lineshape (e.g. had been heterogeneously broadened by interactions with the surrounding environment).  As above, move the sliders and note what happens to the shape and width of the measured signal when the moving gaussian (your laser pulse) is much wider or much narrower than the stationary pulse (the system lineshape). In this case, the shape of the measured signal (the convolution) won't change but think about how the width of the signal relates to the width of the laser pulse or the system lineshape.
    """)
    return


@app.cell
def _(mo):
    sigma1 = mo.ui.slider(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=0.5,
        label="Gaussian 1 width σ1",
    )

    sigma2 = mo.ui.slider(
        start=0.1,
        stop=5.0,
        step=0.1,
        value=2.0,
        label="Gaussian 2 width σ2",
    )

    tau2 = mo.ui.slider(
        start=-10.0,
        stop=10.0,
        step=0.1,
        value=0.0,
        label="Gaussian 2 centre τ",
    )
    return sigma1, sigma2, tau2


@app.cell
def _(mo, sigma1, sigma2, tau2):
    mo.vstack([
        sigma1,
        sigma2,
        tau2,
    ])
    return


@app.cell
def _(
    GAUSS_COLOUR,
    LOR_COLOUR,
    OVERLAP_COLOUR,
    gaussian,
    np,
    plt,
    sigma1,
    sigma2,
    t,
    tau2,
):
    # --------------------------------------------------
    # Top-panel data
    # --------------------------------------------------

    tm= np.linspace(-15, 15, 4000)

    G1 = gaussian(t, sigma1.value, 1)

    G2 = gaussian(
        tm- tau2.value,
        sigma2.value,
        1,
    )

    overlap2 = np.minimum(G1, G2)

    current_conv2 = np.trapezoid(G1 * G2, tm)

    # --------------------------------------------------
    # Precompute convolution trace
    # --------------------------------------------------

    tau2_grid = np.linspace(-10, 10, 300)

    conv_values2 = []

    for tau_j in tau2_grid:

        G_i_2 = gaussian(
            tm- tau_j,
            sigma2.value,
            1,
        )

        conv_values2.append(
            np.trapezoid(G1 * G_i_2, tm)
        )

    conv_values2 = np.array(conv_values2)

    # Find which part of the trace should be drawn
    mask2 = tau2_grid <= tau2.value

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig2, (ax3, ax4) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # --------------------------------------------------
    # Top panel
    # --------------------------------------------------

    ax3.plot(
        t,
        G1,
        color=LOR_COLOUR,
        lw=3,
        label="Gaussian 1",
    )

    ax3.plot(
        t,
        G2,
        color=GAUSS_COLOUR,
        lw=3,
        label="Gaussian 2",
    )

    ax3.fill_between(
        t,
        0,
        overlap2,
        color=OVERLAP_COLOUR,
        alpha=0.8,
    )

    ax3.set_xlim(-15, 15)

    ax3.set_ylabel("Amplitude")

    ax3.set_title(
        f"Current convolution value = {current_conv2:.4f}"
    )

    ax3.legend(frameon=False)

    # --------------------------------------------------
    # Bottom panel
    # --------------------------------------------------

    ax4.plot(
        tau2_grid[mask2],
        conv_values2[mask2],
        color="black",
        lw=2.5,
    )

    ax4.plot(
        tau2.value,
        current_conv2,
        "o",
        color="black",
        markersize=7,
    )

    ax4.set_xlim(
        -15,
        15,
    )

    ax4.set_ylim(
        0,
        1.05 * conv_values2.max(),
    )

    ax4.axvline(
        tau2.value,
        color="grey",
        ls="--",
        alpha=0.5,
    )

    ax4.set_xlabel("Gaussian2 centre τ")
    ax4.set_ylabel("Convolution")

    fig2.tight_layout()

    fig2
    return


@app.cell
def _(app):
    if __name__ == "__main__":
        app.run()
    return


if __name__ == "__main__":
    app.run()
