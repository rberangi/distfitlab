"""distfitlab - interactive distribution fitting in Jupyter.

Fit probability distributions to your data, compare them by the largest gap between
the empirical and theoretical CDF, and inspect the data itself - all from one widget
UI with two halves:

    continuous  18 distributions (Normal, Gamma, Weibull, GEV, Student-T, ...)
    discrete     5 count distributions (Poisson, Binomial, Geometric,
                 Negative Binomial, Zero-Inflated Poisson)

Usage:

    from distfitlab import main
    main()                 # opens on the continuous app
    main("Discrete")       # opens on the discrete app

Building the widgets takes a couple of seconds, so it is deferred until `main` is
actually looked up rather than done on `import distfitlab`.
"""

__version__ = "0.1.0"
__all__ = ["main"]


def __getattr__(name):          # PEP 562: keeps `import distfitlab` cheap
    if name == "main":
        from .app import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
