import json
import numpy as np
from PIL import Image

from segrobust.cli import main


def test_cli_list(capsys):
    assert main(['list']) == 0
    out = capsys.readouterr().out
    assert 'gaussian_noise' in out and 'elastic_transform' in out
    assert 'geometric' in out


def test_cli_corrupt(tmp_path, capsys):
    rng = np.random.RandomState(0)
    img = rng.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    src = tmp_path / 'in.png'
    dst = tmp_path / 'out.png'
    Image.fromarray(img).save(src)
    assert main(['corrupt', str(src), '--name', 'fog', '--severity', '2',
                 '--seed', '5', '--out', str(dst)]) == 0
    out = np.array(Image.open(dst))
    assert out.shape == (32, 32, 3)
    # seeded CLI run is reproducible
    dst2 = tmp_path / 'out2.png'
    main(['corrupt', str(src), '--name', 'fog', '--severity', '2',
          '--seed', '5', '--out', str(dst2)])
    assert np.array_equal(out, np.array(Image.open(dst2)))


def test_cli_report(tmp_path, capsys):
    result = {
        'clean_miou': 0.9,
        'per_corruption': {'fog': {'1': 0.8, '2': 0.7}},
        'mean_corrupt_miou': 0.75,
        'relative_drop': 0.1667,
        'n_images': 4,
        'corruptions': ['fog'],
        'severities': [1, 2],
    }
    p = tmp_path / 'res.json'
    p.write_text(json.dumps(result))
    assert main(['report', str(p)]) == 0
    out = capsys.readouterr().out
    assert '| fog |' in out and '0.9000' in out
