import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    app = mo.App()
    return app, mo, np, plt


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
        label="Gaussian centre",
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
    return


@app.cell
def _(app):
    if __name__ == "__main__":
        app.run()
    return


if __name__ == "__main__":
    app.run()
