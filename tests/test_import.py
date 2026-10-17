"""Test trivial : le paquet et ses modules s'importent."""


def test_import_package():
    import dupire

    assert dupire.__version__ == "0.1.0"


def test_modules_importables():
    from dupire import black_scholes, local_vol, monte_carlo, pde_solver  # noqa: F401
