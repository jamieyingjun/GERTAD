"""Rebuild frozen TEST inputs from GeoLife and prepare pinned encoder sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def checked_bytes(path, expected):
    payload = Path(path).read_bytes()
    if digest(payload) != expected:
        raise RuntimeError('Input integrity check failed: ' + Path(path).name)
    return payload


def prepare_sources():
    sources = json.loads((ROOT / 'config/external_sources.json').read_text(encoding='utf-8'))
    for item in sources:
        target = ROOT / item['path']
        if target.exists():
            checked_bytes(target, item['sha256'])
            continue
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'GERTAD-source-setup'})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        candidates = [payload, payload.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')]
        payload = next((p for p in candidates if digest(p) == item['sha256']), None)
        if payload is None:
            raise RuntimeError('Pinned external source hash mismatch: ' + item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print('Prepared: ' + item['path'], flush=True)


def locate_geolife(requested):
    if requested is not None:
        candidates = [requested]
    else:
        base = ROOT / 'data/geolife'
        candidates = [base / 'Geolife Trajectories 1.3.zip',
                      base / 'Geolife Trajectories 1.3', base]
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.is_file() and candidate.suffix.lower() == '.zip':
            return candidate
        if candidate.is_dir():
            for folder in (candidate / 'Data', candidate):
                if (folder / '002/Trajectory').is_dir():
                    return folder
    raise FileNotFoundError(
        'GeoLife input was not found. Download Geolife Trajectories 1.3.zip from '
        'the official Microsoft page and place it in data/geolife/ beside this '
        'project. Keep the ZIP filename unchanged. See README.md, steps 1-3.')


def selected_points(payload, indices):
    rows = payload.decode('utf-8-sig').splitlines()[6:]
    coordinates, timestamps, dates = [], [], {}
    for index in indices:
        fields = rows[index].split(',')
        coordinates.append([struct.unpack('<f', struct.pack('<f', float(value)))[0]
                            for value in fields[:2]])
        day, clock = fields[5].strip(), fields[6].strip()
        if day not in dates:
            dates[day] = int(datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        seconds = dates[day] + int(clock[:2])*3600 + int(clock[3:5])*60 + int(clock[6:8])
        timestamps.append(float(seconds))
    return coordinates, timestamps


def rebuild_worlds(source, manifest):
    recipe = json.loads(checked_bytes(ROOT / 'config/geolife_windows.json',
                                     manifest['sha256']['config/geolife_windows.json']))
    candidates = json.loads(gzip.decompress(checked_bytes(ROOT / 'data/map_candidates.json.gz',
                                        manifest['sha256']['data/map_candidates.json.gz'])))
    entities = {item['context_id']: item['entities'] for item in candidates}
    if len(entities) != len(recipe['windows']):
        raise RuntimeError('The frozen trajectory recipe and candidate contexts differ.')
    bundle = zipfile.ZipFile(source) if source.is_file() else None
    try:
        members = {}
        if bundle is not None:
            for item in bundle.infolist():
                parts = item.filename.replace('\\', '/').split('/')
                if 'Data' in parts and not item.is_dir():
                    relative = '/'.join(parts[parts.index('Data')+1:])
                    if relative in recipe['trajectories']:
                        if relative in members:
                            raise RuntimeError('Duplicate trajectory in GeoLife archive: ' + relative)
                        members[relative] = item
        payloads = {}
        for relative, identity in recipe['trajectories'].items():
            if bundle is None:
                payload = (source / relative).read_bytes()
            else:
                if relative not in members:
                    raise FileNotFoundError('Required GeoLife trajectory is missing: Data/' + relative)
                item = members[relative]
                if item.file_size != identity['bytes']:
                    raise RuntimeError('GeoLife trajectory size differs: ' + relative)
                payload = bundle.read(item)
            if digest(payload) != identity['sha256']:
                raise RuntimeError('GeoLife trajectory hash differs: ' + relative +
                                   '. Use the original GeoLife 1.3 files without editing them.')
            payloads[relative] = payload
        worlds = []
        for window in recipe['windows']:
            coordinates, timestamps = selected_points(payloads[window['trajectory']], window['point_indices'])
            worlds.append(dict(context_id=window['context_id'], coordinates=coordinates,
                               timestamps=timestamps, entities=entities[window['context_id']]))
        raw = json.dumps(worlds, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        output = io.BytesIO()
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as stream:
            stream.write(raw)
        payload = output.getvalue()
        if digest(payload) != manifest['sha256']['data/worlds.json.gz']:
            raise RuntimeError('Reconstructed inputs differ from the frozen benchmark; no input file was written.')
        print('Rebuilt 271 TEST contexts from 190 verified GeoLife trajectory files.', flush=True)
        return payload
    finally:
        if bundle is not None:
            bundle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geolife', type=Path, help='Optional GeoLife 1.3 ZIP, extracted dataset folder or Data folder.')
    parser.add_argument('--worlds', type=Path, help='Optional existing prepared input, verified against the frozen SHA-256.')
    parser.add_argument('--rebuild', action='store_true', help='Verify reconstruction even when the prepared input exists.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    target = ROOT / 'data/worlds.json.gz'
    expected = manifest['sha256']['data/worlds.json.gz']
    if target.exists():
        checked_bytes(target, expected)
    if args.worlds is not None:
        payload = checked_bytes(args.worlds, expected)
    elif not target.exists() or args.rebuild:
        source = locate_geolife(args.geolife)
        print('Preparing frozen GeoLife TEST inputs...', flush=True)
        payload = rebuild_worlds(source, manifest)
    else:
        payload = None
    if payload is not None:
        temporary = target.with_suffix(target.suffix + '.partial')
        temporary.write_bytes(payload)
        temporary.replace(target)
    print('Trajectory input verified: ' + expected, flush=True)
    prepare_sources()
    print('Input preparation completed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
        raise SystemExit('Preparation failed: ' + str(error))
