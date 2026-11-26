"""Command-line parameters merged with the cfg_args a training run left behind."""

import sys
from argparse import ArgumentParser

from arguments import ModelParams, ReflectionOptimizationParams, get_combined_args


def test_parameters_missing_from_an_old_config_take_their_defaults(tmp_path, monkeypatch):
    # A Phase I config written before the Fresnel parameters existed.
    (tmp_path / 'interface').mkdir()
    (tmp_path / 'interface' / 'cfg_args').write_text("Namespace(source_path='/data/scene', model_path='old', resolution=4, eval=True, sh_degree=2, ode_chunk=5000)")

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    opt = ReflectionOptimizationParams(parser)
    monkeypatch.setattr(sys, 'argv', ['train_reflection.py', '-m', str(tmp_path), '--iterations', '123'])
    args = get_combined_args(parser, cfg_subdir='interface')

    dataset = model.extract(args)
    assert dataset.model_path == str(tmp_path)  # the command line wins over the config
    assert (dataset.resolution, dataset.eval, dataset.sh_degree, dataset.ode_chunk) == (4, True, 2, 5000)  # from the config
    assert (dataset.eta_air, dataset.eta_glass, dataset.optical_correction) == (1.0, 1.5, False)  # defaults
    assert dataset.static_interface is False and dataset.clean_images == 'clean_image'
    assert opt.extract(args).iterations == 123
