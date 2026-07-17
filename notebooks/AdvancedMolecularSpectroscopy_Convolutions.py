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

    return FuncAnimation, HTML, mo, np, plt


@app.cell(hide_code=True)
def _(plt):
    # APPEARANCE SETTINGS
    # ============================================================

    LOR_COLOUR = (187 / 255, 85 / 255, 102 / 255)
    GAUSS_COLOUR = (0 / 255, 68 / 255, 136 / 255)
    OVERLAP_COLOUR = (221 / 255, 170 / 255, 51 / 255)
    CONV_COLOUR = "black"

    LINEWIDTH_DISTRIBUTIONS = 3.0
    LINEWIDTH_CONVOLUTION = 2.5

    FONT_FAMILY = "DejaVu Sans"
    TITLE_SIZE = 14
    LABEL_SIZE = 14
    TICK_SIZE = 12

    plt.rcParams["font.family"] = FONT_FAMILY
    return (
        CONV_COLOUR,
        GAUSS_COLOUR,
        LABEL_SIZE,
        LINEWIDTH_CONVOLUTION,
        LINEWIDTH_DISTRIBUTIONS,
        LOR_COLOUR,
        OVERLAP_COLOUR,
        TICK_SIZE,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below are definitions of two types of distributions (or functions) that we want to experiment with convolving: a lorentzian and a gaussian. The formulae for these functions should look familiar from the lecture slides! If you would like to play with convolutions of other functions, feel free to add your own additional function definitions in the cell below.
    """)
    return


@app.cell
def _(np):
    def lorentzian(t, gamma):
        """
        Unit-area Lorentzian.
        """
        return (2 * gamma )/(t**2 + gamma**2)


    def gaussian(t, sigma, gamma):
        """
        Gaussian scaled to have same peak height as Lorentzian.
        """

        lorentzian_peak = 2.0 / gamma

        return lorentzian_peak * (np.exp(-t**2 / (2 * sigma**2)))

    return gaussian, lorentzian


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the cell below, you can change the parameters for the two distributions. Increasing $\gamma$ (gamma) will increase the width of the Lorenztian distribution. Increasing $\sigma$ (sigma) will increase the width of the Gaussian distribution.

    You are unlikely to need to change the range or number of points on the t (time) and $\tau$ (shift) axes but you are free to do so if you wish.
    """)
    return


@app.cell
def _(np):
    # Adjust the width of the Lorentzian
    gamma = 0.5

    # Adjust the width of the Gaussian
    sigma = 2.0

    # Set the range over t values over which the distributions will be computed
    t = np.linspace(-15, 15, 1000)

    # Set the range of shift values for the animation
    taus = np.linspace(-10, 10, 50)
    return gamma, sigma, t, taus


@app.cell(hide_code=True)
def _(
    CONV_COLOUR,
    FuncAnimation,
    GAUSS_COLOUR,
    HTML,
    LABEL_SIZE,
    LINEWIDTH_CONVOLUTION,
    LINEWIDTH_DISTRIBUTIONS,
    LOR_COLOUR,
    OVERLAP_COLOUR,
    TICK_SIZE,
    gamma,
    gaussian,
    lorentzian,
    np,
    plt,
    sigma,
    t,
    taus,
):
    # ============================================================
    # CALCULATION AND ANIMATION
    # ============================================================
    # Define the timestep
    dt = t[1] - t[0]

    # Calculate the static distributions centred at t=0
    L = lorentzian(t, gamma)
    G = gaussian(t, sigma, gamma)

    # ============================================================
    # PRECOMPUTE CONVOLUTION VALUES
    # ============================================================

    conv_values = []

    for tau in taus:
        G_shifted = gaussian(tau - t, sigma, gamma)
        conv = np.trapezoid(L * G_shifted, t)
        conv_values.append(conv)

    conv_values = np.array(conv_values)

    # ==================*==================================*======
    # FIGURE
    # ================*==================================*========

    fig = plt.figure(figsize=(10, 6))

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[3, 1.5],
        hspace=0.3
    )

    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # ------------------------------------------------------------
    # TOP PANEL
    # ------------------------------------------------------------

    ax_top.set_xlim(t.min(), t.max())
    ax_top.set_ylim(0, max(L) * 1.15)

    ax_top.set_xlabel(r"$\omega$", fontsize=LABEL_SIZE)
    ax_top.set_ylabel("Amplitude", fontsize=LABEL_SIZE)

    ax_top.tick_params(labelsize=TICK_SIZE)

    lor_line, = ax_top.plot(
        t,
        L,
        color=LOR_COLOUR,
        lw=LINEWIDTH_DISTRIBUTIONS,
        label="Lorentzian",
    )

    gauss_line, = ax_top.plot(
        [],
        [],
        color=GAUSS_COLOUR,
        lw=LINEWIDTH_DISTRIBUTIONS,
        label="Gaussian",
    )

    lor_line2, = ax_bottom.plot(
        t,
        G,
        color=GAUSS_COLOUR, alpha=0.5,
        lw=LINEWIDTH_DISTRIBUTIONS,
        label="Gaussian",
    )

    overlap_fill = None

    # ----------------------------------------------------------
    # BOTTOM PANEL
    # -----------------------------------------------------------

    ax_bottom.set_xlim(taus.min(), taus.max())
    ax_bottom.set_ylim(0, conv_values.max() * 1.1)

    ax_bottom.set_xlabel(r"$\omega$'", fontsize=LABEL_SIZE)
    ax_bottom.set_ylabel("Convolution", fontsize=LABEL_SIZE)

    ax_bottom.tick_params(labelsize=TICK_SIZE)

    conv_line, = ax_bottom.plot(
        [],
        [],
        color=CONV_COLOUR,
        lw=LINEWIDTH_CONVOLUTION,
    )

    current_point, = ax_bottom.plot(
        [],
        [],
        marker="o",
        color=CONV_COLOUR,
        markersize=6,
    )

    # ============================================================
    # ANIMATION FUNCTIONS
    # ============================================================

    def init():
        gauss_line.set_data([], [])
        conv_line.set_data([], [])
        current_point.set_data([], [])
        return gauss_line, conv_line, current_point


    def update(frame):

        global overlap_fill

        tau = taus[frame]

        G_shifted = gaussian(tau - t, sigma, gamma)

        gauss_line.set_data(t, G_shifted)

        # Remove previous shaded region
        if overlap_fill is not None:
            overlap_fill.remove()

        overlap = np.minimum(L, G_shifted)

        overlap_fill = ax_top.fill_between(
            t,
            0,
            overlap,
            color=OVERLAP_COLOUR,
            alpha=0.9,
        )

        conv_line.set_data(
            taus[: frame + 1],
            conv_values[: frame + 1],
        )

        current_point.set_data(
            [tau],
            [conv_values[frame]],
        )

        artists = (
            gauss_line,
            conv_line,
            current_point,
            overlap_fill,
        )

        return artists


    # ============================================================
    # CREATE ANIMATION
    # ============================================================

    anim = FuncAnimation(
        fig,
        update,
        frames=len(taus),
        init_func=init,
        interval=40,
        blit=False,
    )

    HTML(anim.to_html5_video())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
