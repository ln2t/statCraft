import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nilearn.image import resample_to_img

from statcraft.core.geometry import GeometryAudit
from statcraft.pipeline import StatCraftPipeline
from statcraft.cli import create_parser


def save(path, data, shift=0):
    affine = np.eye(4)
    affine[0, 3] = shift
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), path)
    return path


def test_audit_includes_excluded_unreadable_and_unpaired_images(tmp_path):
    first = save(tmp_path / 'first.nii.gz', np.ones((4, 4, 4)))
    other = save(tmp_path / 'different.nii.gz', np.ones((5, 4, 4)), shift=2)
    broken = tmp_path / 'broken.nii.gz'
    broken.write_bytes(b'not nifti')
    audit = GeometryAudit(tmp_path / 'out')
    infos = [{'path': p} for p in (first, other, broken)]
    audit.collect(infos)
    valid, invalid = audit.validate(infos[:2])
    assert len(valid) == len(invalid) == 1
    records = json.loads((tmp_path / 'out/geometry.json').read_text())['images']
    assert len(records) == 3
    assert records[1]['origin_displacement_mm'] == 2
    assert records[1]['max_corner_displacement_mm'] == 2
    assert records[1]['status'] == 'excluded (geometry)'
    html = (tmp_path / 'out/geometry.html').read_text()
    assert all(str(path) in html for path in (first, other, broken))
    assert 'median 1' in html


def test_4d_reference_resampling_preserves_world_signal_and_original(tmp_path):
    values = np.indices((7, 7, 7)).sum(axis=0).astype(float)
    original = save(tmp_path / 'sub-01_stat.nii.gz', values, shift=1)
    before = original.read_bytes()
    reference = save(tmp_path / 'reference.nii.gz', np.zeros((5, 5, 5, 2)))
    audit = GeometryAudit(tmp_path / 'out', reference, True)
    info = {'path': original}
    valid, invalid = audit.validate([info])
    assert not invalid
    actual = nib.load(valid[0]['path'])
    expected = resample_to_img(nib.load(original), audit.reference, interpolation='continuous')
    np.testing.assert_allclose(actual.get_fdata(), expected.get_fdata())
    assert actual.shape == (5, 5, 5)
    assert original.read_bytes() == before
    assert info['original_path'] == original
    assert audit.records[str(original)]['status'] == 'resampled'
    audit.validate(valid)  # Re-validation must not resample an already prepared image.
    assert info['original_path'] == original
    mask = nib.Nifti1Image((values % 2).astype(np.uint8), nib.load(original).affine)
    mask_result = audit.mask(mask, audit.reference)
    assert set(np.unique(mask_result.get_fdata())) <= {0, 1}


@pytest.mark.parametrize('route', ['one-sample', 'two-sample', 'paired', 'combine', 'legacy'])
def test_routes_harmonize_before_glm_and_keep_sibling_masks(tmp_path, route):
    data = tmp_path / 'inputs'
    data.mkdir()
    (data / 'dataset_description.json').write_text(json.dumps({'Name': 'geometry', 'BIDSVersion': '1.8.0'}))
    (data / 'participants.tsv').write_text('participant_id\n' + ''.join(
        f'sub-{subject:02}\n' for subject in range(1, 5)))
    reference = save(tmp_path / 'ref.nii.gz', np.zeros((4, 4, 4, 2)))
    for subject in range(1, 5):
        for condition in ('A', 'B'):
            shift = 0.2 if condition == 'B' or subject == 4 else 0
            entity = 'ses' if route == 'legacy' else 'desc'
            values = np.random.default_rng(subject + (condition == 'B')).normal(size=(4, 4, 4))
            save(data / f'sub-{subject:02}_{entity}-{condition}_stat.nii.gz', values, shift)
        save(data / f'sub-{subject:02}_desc-brain_mask.nii.gz', np.ones((4, 4, 4)))
    patterns = {c: f'*desc-{c}_stat.nii.gz' for c in ('A', 'B')}
    config = dict(analysis_type='paired' if route in ('combine', 'legacy') else route,
                  reference_image=str(reference), harmonize_affines=True,
                  inference={'corrections': []}, output={'generate_report': False}, verbose=0)
    if route in ('paired', 'combine'):
        config['paired_test'] = {'sample_patterns': patterns, 'combine': 'A-B' if route == 'combine' else None}
    elif route == 'legacy':
        config['paired_test'] = {'pair_by': 'session', 'condition1': 'A', 'condition2': 'B'}
    elif route == 'two-sample':
        config['sample_patterns'] = patterns
    pipeline = StatCraftPipeline(data, tmp_path / 'out', [], config=config)
    pipeline.annotate_clusters = lambda: {}
    result = pipeline.run(sample_patterns=patterns if route in ('paired', 'combine', 'two-sample') else None,
                          pattern='*ses-*_stat.nii.gz' if route == 'legacy' else patterns['A'])
    assert not pipeline._invalid_images
    assert len(pipeline.geometry.records) == (4 if route == 'one-sample' else 8)
    assert any(row['status'] == 'resampled' for row in pipeline.geometry.records.values())
    assert pipeline._inference_mask_metadata['MNI152MaskSource'] is None
    for info in pipeline._images:
        image = nib.load(info['path'])
        np.testing.assert_array_equal(image.affine, np.eye(4))
    assert result['saved_files']['inference_mask'].exists()
    if route == 'one-sample':
        report = pipeline.generate_report()
        assert 'Image geometry (all collected images)' in report.read_text()
        section = next(s for s in pipeline.report.sections if s['title'].startswith('Image geometry'))
        assert 'Collected: 4' in section['content']
        assert 'resampled' in section['content']


def test_cli_geometry_arguments():
    args = create_parser().parse_args(['input', 'output', 'group', '--harmonize-affines',
                                      '--reference-image', '/opt/fsl/data/standard/MNI152_T1_2mm.nii.gz'])
    assert args.harmonize_affines
    assert args.reference_image.endswith('MNI152_T1_2mm.nii.gz')


def test_config_only_cli_preserves_geometry_options_and_cli_reference_override(tmp_path, monkeypatch):
    import sys
    import yaml
    from unittest.mock import MagicMock
    import statcraft.cli as cli
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'analysis_type': 'one-sample', 'harmonize_affines': True,
                                     'reference_image': '/configured/reference.nii.gz'}))
    constructor = MagicMock()
    constructor.return_value.run.return_value = {'saved_files': {}}
    monkeypatch.setattr(cli, 'StatCraftPipeline', constructor)
    for extra, expected in (([], '/configured/reference.nii.gz'),
                            (['--reference-image', '/override.nii.gz'], '/override.nii.gz')):
        monkeypatch.setattr(sys, 'argv', ['statcraft', str(tmp_path), str(tmp_path / 'out'),
                                         'group', '--config', str(config)] + extra)
        cli.main()
        actual = constructor.call_args.kwargs['config']
        assert actual.get('harmonize_affines') is True
        assert actual.get('reference_image') == expected


def test_full_html_report_includes_geometry_exclusions(tmp_path):
    data = tmp_path / 'inputs'
    data.mkdir()
    reference = save(tmp_path / 'ref.nii.gz', np.zeros((4, 4, 4)))
    mask = save(tmp_path / 'brain.nii.gz', np.ones((4, 4, 4)))
    for subject in range(1, 6):
        save(data / f'sub-{subject:02}_stat.nii.gz',
             np.random.default_rng(subject).normal(size=(4, 4, 4)), shift=2 if subject == 5 else 0)
    pipeline = StatCraftPipeline(data, tmp_path / 'out', [], config={
        'analysis_type': 'one-sample', 'reference_image': str(reference),
        'inference_mask': str(mask), 'output': {'generate_report': False},
        'inference': {'corrections': []}, 'verbose': 0})
    pipeline.annotate_clusters = lambda: {}
    pipeline.run(pattern='*_stat.nii.gz')
    assert len(pipeline._images) == 4
    report = pipeline.generate_report().read_text()
    assert 'Collected: 5; matching: 4; mismatched: 1' in report
    assert 'sub-05_stat.nii.gz' in report
    assert 'excluded (geometry)' in report
    assert 'median 0, max 2' in report
