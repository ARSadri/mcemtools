#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest

from click.testing import CliRunner

import mcemtools
import mcemtools.denoise.denoise_4DSTEM.denoise_4DSTEM as mcemdenoiser

from mcemtools import cli

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    #assert 'mcemtools.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

def test_mcemtools_denoise_4DSTEM():
    
    #data_4DSTEM = np.load('data_from_Py4DSTEM.npy')
    data_4DSTEM = 10 + np.random.randn(256, 256, 64, 64)
    denoised_data = mcem_denoiser(data_4DSTEM)
    