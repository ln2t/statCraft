"""Audit original NIfTI geometry and optionally resample onto one explicit grid."""
import hashlib
import html
import itertools
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

logger = logging.getLogger(__name__)


class GeometryAudit:
    tolerance = 1e-5

    def __init__(self, output_dir, reference_image=None, harmonize=False):
        self.output_dir = Path(output_dir)
        self.harmonize = harmonize
        self.reference = None
        self.reference_path = None
        self.records = {}
        if reference_image:
            self._set_reference(Path(reference_image).expanduser())

    def _set_reference(self, path):
        image = nib.load(path, mmap=True)
        if image.ndim not in (3, 4) or any(n < 1 for n in image.shape):
            raise ValueError(f"Reference image must be a nonempty 3D or 4D NIfTI: {path}")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
            raise ValueError(f"Reference image has a nonfinite or singular affine: {path}")
        # Only the spatial grid is used; no temporal averaging or volume selection.
        self.reference = nib.Nifti1Image(np.zeros(image.shape[:3], dtype=np.uint8), image.affine)
        self.reference_path = str(path.resolve())

    def matches(self, image):
        return (image.ndim == 3 and image.shape == self.reference.shape and
                np.allclose(image.affine, self.reference.affine, atol=self.tolerance, rtol=0))

    def collect(self, images):
        for info in images:
            path = Path(info.get('original_path', info['path'])).resolve()
            key = str(path)
            if key in self.records:
                continue
            record = {'path': key, 'status': 'not validated'}
            self.records[key] = record
            try:
                image = nib.load(path, mmap=True)
                if self.reference is None and image.ndim == 3:
                    self._set_reference(path)
                record.update(shape=list(image.shape), affine=image.affine.tolist(),
                              voxel_sizes_mm=list(nib.affines.voxel_sizes(image.affine)),
                              orientation=''.join(c or '?' for c in nib.aff2axcodes(image.affine)))
                if image.ndim != 3:
                    raise ValueError(f"Expected a 3D map, got {image.shape}")
                if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
                    raise ValueError('Nonfinite or singular affine')
                record['matches_reference'] = self.matches(image)
                record['max_affine_difference'] = float(np.abs(image.affine-self.reference.affine).max())
                record['origin_displacement_mm'] = float(np.linalg.norm(
                    image.affine[:3, 3]-self.reference.affine[:3, 3]))
                corners = np.array(list(itertools.product(*[(0, n-1) for n in image.shape])))
                delta = nib.affines.apply_affine(image.affine, corners) - nib.affines.apply_affine(
                    self.reference.affine, corners)
                record['max_corner_displacement_mm'] = float(np.linalg.norm(delta, axis=1).max())
                record['status'] = 'matching' if record['matches_reference'] else 'mismatched'
            except Exception as error:
                if 'affine' in record and not np.isfinite(record['affine']).all():
                    record['affine'] = repr(record['affine'])
                record['error'] = str(error)
                record['status'] = 'unreadable/invalid'
        if self.records:
            self.save()

    def validate(self, images):
        self.collect(images)
        valid, invalid = [], []
        for info in images:
            original = Path(info.get('original_path', info['path'])).resolve()
            record = self.records[str(original)]
            if 'error' in record:
                invalid.append({**info, 'reason': record['error']})
                continue
            if record['matches_reference']:
                valid.append(info)
            elif not self.harmonize:
                record['status'] = 'excluded (geometry)'
                reason = ('Shape mismatch with reference image' if tuple(record['shape']) != self.reference.shape
                          else 'Affine mismatch with reference image')
                invalid.append({**info, 'reason': reason})
            else:
                # Keep original names/entities, isolating equal filenames by source path.
                digest = hashlib.sha256(str(original).encode()).hexdigest()[:16]
                destination = self.output_dir / 'harmonized' / digest / original.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not info.get('original_path'):
                    source = nib.load(original, mmap=True)
                    result = resample_to_img(source, self.reference, interpolation='continuous')
                    nib.save(result, destination)
                    info['original_path'] = original
                    info['path'] = destination
                record['status'] = 'resampled'
                record['harmonized_path'] = str(destination.resolve())
                valid.append(info)
        for info in invalid:
            logger.warning('Excluded %s: %s', info['path'], info['reason'])
        self.save()
        return valid, invalid

    def mask(self, image, reference):
        """Resample anatomical/normalization masks only when explicitly enabled."""
        if image.ndim != 3:
            raise ValueError('Mask must be a 3D image')
        values = np.asarray(image.dataobj)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Mask must be finite and nonnegative')
        if self.harmonize and (image.shape != reference.shape or not np.allclose(
                image.affine, reference.affine, atol=self.tolerance, rtol=0)):
            return resample_to_img(image, reference, interpolation='nearest')
        return image

    def html(self):
        escape = lambda value: html.escape(str(value))
        rows = list(self.records.values())
        matching = sum(r.get('matches_reference', False) for r in rows)
        readable = [r for r in rows if 'max_affine_difference' in r]
        content = (f'<p>Reference: {escape(self.reference_path)}; tolerance: {self.tolerance:g} '
                   'absolute, relative tolerance 0. Original geometry before exclusions/resampling.</p>'
                   f'<p>Collected: {len(rows)}; matching: {matching}; mismatched: '
                   f'{len(readable)-matching}; unreadable/invalid: {len(rows)-len(readable)}. '
                   f'Harmonization enabled: {self.harmonize}.</p>')
        for metric in ('max_affine_difference', 'origin_displacement_mm', 'max_corner_displacement_mm'):
            values = [r[metric] for r in readable]
            if values:
                content += (f'<p>{metric} (all readable 3D maps): min {min(values):.6g}, '
                            f'median {np.median(values):.6g}, max {max(values):.6g}.</p>')
        content += ('<p>Geometry status describes grid validation only; pairing and other exclusions are listed in the analysis data section.</p>'
                    '<p>Corner displacement compares the same voxel indices under both affines; '
                    'it is a grid discrepancy, not an anatomical registration error. '
                    'Shape differences are listed separately. Resampling does not perform registration.</p>')
        groups = {}
        for row in readable:
            key = json.dumps([row['shape'], row['affine']])
            groups[key] = groups.get(key, 0) + 1
        content += f'<p>Distinct exact shape/affine grids: {len(groups)}; group sizes: {escape(sorted(groups.values(), reverse=True))}.</p>'
        content += '<table><thead><tr>' + ''.join(f'<th>{label}</th>' for label in (
            'Original file', 'Shape', 'Voxel sizes (mm)', 'Orientation', 'Max affine difference',
            'Origin delta (mm)', 'Max corner delta (mm)', 'Geometry status', 'Original affine / error')) + '</tr></thead><tbody>'
        for row in rows:
            values = [row['path'], row.get('shape'), row.get('voxel_sizes_mm'), row.get('orientation'),
                      row.get('max_affine_difference'), row.get('origin_displacement_mm'),
                      row.get('max_corner_displacement_mm'), row['status'], row.get('error', row.get('affine'))]
            content += '<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in values) + '</tr>'
        return content + '</tbody></table>'

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(reference=self.reference_path, harmonize_affines=self.harmonize,
                       tolerance=self.tolerance, interpolation='continuous',
                       mask_interpolation='nearest', images=list(self.records.values()))
        (self.output_dir / 'geometry.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
        (self.output_dir / 'geometry.html').write_text(
            '<!doctype html><html><head><meta charset="utf-8"><title>StatCraft image geometry</title>'
            '<style>table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:5px;overflow-wrap:anywhere}</style>'
            '</head><body><h1>Image geometry (all collected images)</h1>' + self.html() + '</body></html>')
