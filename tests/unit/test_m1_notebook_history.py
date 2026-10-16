"""Regression checks for the preserved M1 notebook, not the active launcher."""

import ast
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def notebook_cells():
    path = Path(__file__).resolve().parents[2] / (
        "notebooks/archive/03_m1_and_preference_before_reorganization_20261013.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    }


def test_archived_workflow_keeps_its_original_cell_order():
    path = Path(__file__).resolve().parents[2] / (
        "notebooks/archive/03_m1_and_preference_before_reorganization_20261013.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert [cell["id"] for cell in cells[:8]] == [
        "gpu-gate-intro", "gpu-gate-mount", "gpu-gate-bootstrap", "gpu-gate-prepare",
        "gpu-gate-m2", "gpu-gate-m3", "gpu-gate-history", "portable-intro",
    ]
    cells = cells[7:]
    assert [cell["id"] for cell in cells[:12]] == [
        "portable-intro", "portable-mount", "portable-source", "portable-data",
        "portable-epoch-audit", "portable-epoch-selection", "portable-validation-setup",
        "portable-full-validation", "preference-stage-intro", "preference-freeze-m1",
        "preference-generate-main", "validation-intro",
    ]
    assert "Lịch sử — không chạy lại" in "".join(cells[11]["source"])
    assert "Không bấm Chạy tất cả" in "".join(cells[0]["source"])
    assert all(cell["id"] != "portable-m1" for cell in cells[:12])


def test_preference_stage_freezes_m1_before_any_gpu_generation():
    cells = notebook_cells()
    setup = cells['preference-freeze-m1']
    run = cells['preference-generate-main']
    ast.parse(setup)
    ast.parse(run)
    assert 'scripts.freeze_m1_selection' in setup
    assert 'frozen_checkpoint3683_v2' in setup and 'training/checkpoint-3683' in setup
    assert 'source/preference_stage_v2.json' in setup
    assert 'cuda' not in setup and 'pip' not in setup and 'human-review' not in setup
    assert '2000' in run and 'question_pool_v1' in run
    assert "'--m1-selection-sha256', FROZEN_M1_SHA" in run
    assert "'--resume'" in run and "log.flush()" in run
    assert run.index('torch.cuda.is_available()') < run.index('subprocess.run')
    assert 'train_preference' not in run and 'train_dpo' not in run


def candidate_version_function(packages):
    tree = ast.parse(notebook_cells()['preference-generate-main'])
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'installed_candidate_versions')
    expected = next(node.value for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name)
                            and target.id == 'expected_candidate_versions'
                            for target in node.targets))
    scope = {
        'candidate_packages': packages,
        'expected_candidate_versions': ast.literal_eval(expected),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), function.name, 'exec'), scope)
    return scope[function.name], scope['expected_candidate_versions']


@pytest.mark.parametrize('missing', [None, 'peft', 'bitsandbytes'])
def test_preference_bootstrap_reports_absent_packages_without_raising(missing):
    class MissingPackage(Exception):
        pass

    queried = []
    installed = {
        'torch': '2.11.0+cu130', 'transformers': '4.57.6',
        'peft': '0.21.0', 'bitsandbytes': '0.50.2', 'pillow': '11.3.0',
    }

    def version(name):
        queried.append(name)
        if name == missing:
            raise MissingPackage(name)
        return installed[name]

    function, expected = candidate_version_function(SimpleNamespace(
        version=version, PackageNotFoundError=MissingPackage,
    ))
    result = function()
    assert expected == installed
    assert result == {name: None if name == missing else value
                      for name, value in installed.items()}
    assert sorted(queried) == sorted(installed)


def test_preference_bootstrap_preserves_unexpected_metadata_errors():
    class MissingPackage(Exception):
        pass

    def version(name):
        raise RuntimeError('metadata storage failed')

    function, _ = candidate_version_function(SimpleNamespace(
        version=version, PackageNotFoundError=MissingPackage,
    ))
    with pytest.raises(RuntimeError, match='metadata storage failed'):
        function()


def test_preference_bootstrap_is_independent_of_historical_validation_setup():
    run = notebook_cells()['preference-generate-main']
    tree = ast.parse(run)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {'EVAL_ENV', 'EVAL_REPO', 'M0_CACHE', 'cache_manifest', 'eval_packages'}
    assert 'Restore the inference environment in cell 6' not in run
    assert 'installed_candidate_versions()' in run
    assert "'torch==2.11.0+cu130'" in run
    assert "'torchvision==0.26.0+cu130'" in run
    assert 'https://download.pytorch.org/whl/cu130' in run
    assert run.index("'install'") < run.index('scripts.verify_coco_bundle')
    assert run.index("'install'") < run.index('scripts.generate_preference_candidates')
    assert 'Restart' in run or 'restart' in run
    assert 'torch.__version__' in run


def test_preference_fixed_inputs_are_checked_before_bootstrap_and_extraction():
    run = notebook_cells()['preference-generate-main']
    tree = ast.parse(run)
    version_call = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'installed_candidate_versions')
    first_subprocess = min(node.lineno for node in ast.walk(tree)
                           if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute)
                           and isinstance(node.func.value, ast.Name)
                           and node.func.value.id == 'subprocess')
    for digest in (
        '4bedfa0cf41ad84a5552ea6b923a85d9a4956fae4c9969f64bc54844d0b2d1ab',
        'd3c74ac0be475952c7069b573478980af259f1cea30a0bbd0859f6af4a018262',
        'db1f26a79a025a2422c9768a8231cfd00dbe4d053683973a216728c551b1d534',
    ):
        node = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and node.value == digest)
        assert node.lineno < version_call.lineno
        assert node.lineno < first_subprocess
    prefix = run[:sum(len(line) for line in run.splitlines(keepends=True)
                      [:version_call.lineno - 1])]
    assert "'questions.jsonl'" in prefix and "'manifest.json'" in prefix
    assert 'FROZEN_M1.read_bytes()' in prefix
    assert '.hexdigest()' in prefix


@pytest.mark.parametrize('fault', [
    None, 'm1_selection', 'm1_selection_bytes', 'questions.jsonl', 'manifest.json',
])
def test_preference_changed_fixed_inputs_abort_before_package_query(tmp_path, fault):
    source = ast.parse(notebook_cells()['preference-generate-main'])
    prefix = []
    for node in source.body:
        if any(isinstance(item, ast.Call)
               and isinstance(item.func, ast.Name)
               and item.func.id == 'installed_candidate_versions'
               for item in ast.walk(node)):
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            prefix.append(node)

    fixed = {
        'questions.jsonl': '4bedfa0cf41ad84a5552ea6b923a85d9a4956fae4c9969f64bc54844d0b2d1ab',
        'manifest.json': 'd3c74ac0be475952c7069b573478980af259f1cea30a0bbd0859f6af4a018262',
    }
    pool = tmp_path / 'preference_local/question_pool_v1'
    pool.mkdir(parents=True)
    for name in fixed:
        (pool / name).write_bytes(b'changed' if fault == name else name.encode())
    selection_digest = 'db1f26a79a025a2422c9768a8231cfd00dbe4d053683973a216728c551b1d534'
    selection = tmp_path / 'selection.json'
    selection.write_bytes(b'changed' if fault == 'm1_selection_bytes' else b'selection.json')
    known_digests = {**fixed, 'selection.json': selection_digest}

    def package_query(name):
        pytest.fail('Fixed inputs must be checked before querying installed packages')

    def digest(payload):
        return SimpleNamespace(hexdigest=lambda: known_digests.get(payload.decode(), '0' * 64))

    scope = {
        'ROOT': tmp_path,
        'FROZEN_M1': selection,
        'FROZEN_M1_SHA': '0' * 64 if fault == 'm1_selection' else selection_digest,
        'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        'hashlib': SimpleNamespace(sha256=digest),
        'candidate_packages': SimpleNamespace(version=package_query),
    }
    program = compile(ast.Module(body=prefix, type_ignores=[]), 'candidate_input_guard', 'exec')
    if fault:
        with pytest.raises(AssertionError):
            exec(program, scope)
    else:
        exec(program, scope)
        assert scope['pool'] == pool


@pytest.mark.historical
def test_preference_addition_does_not_rewrite_executed_code_history():
    root = Path(__file__).resolve().parents[2]
    history = root / ('artifacts/colab_history/20261007/m1_checkpoint1200_validation/'
                      '03_validation_complete.ipynb')
    if not history.is_file():
        pytest.skip('Executed history is not part of the source release')
    previous = json.loads(history.read_text(encoding='utf-8'))
    current = notebook_cells()
    old_code = [''.join(cell['source']) for cell in previous['cells']
                if cell['cell_type'] == 'code']
    preserved_code = [source for name, source in current.items()
                      if name not in {'preference-freeze-m1', 'preference-generate-main'}
                      and not name.startswith('gpu-gate-')]
    assert preserved_code == old_code


def test_full_validation_has_separate_pinned_runtime_and_verified_cache():
    cells = notebook_cells()
    setup, run = cells['portable-validation-setup'], cells['portable-full-validation']
    assert "selected_m1['status'] == 'passed'" in setup
    assert "source_20261002_062529_455947" in setup and 'EVAL_SHA256' in setup
    assert "cache_manifest['inputs']['versions'] == expected_versions" in setup
    assert "cache_manifest['inputs']['source_files']" in setup
    assert '2.11.0+cu130' in setup and '0.21.0' in setup
    assert 'not cell 2' in setup and "'-e'" not in setup
    assert "'--m0-predictions', str(M0_CACHE)" in run
    assert "'--resume'" in run and "'--checkpoint', str(adapter)" in run
    assert "selected_m1['selected_adapter_sha256']" in run
    assert "comparison['n'] == 1407" in run
    assert "comparison['expected_split'] == 'validation'" in run
    assert "'--split', 'test'" not in run
    runner = Path(__file__).resolve().parents[2] / 'scripts/run_baseline_validation.py'
    assert hashlib.sha256(runner.read_bytes()).hexdigest() in run


def validation_function(name):
    tree = ast.parse(notebook_cells()['portable-full-validation'])
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {'hashlib': hashlib, 'Path': Path, 'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), name, 'exec'), scope)
    return scope


@pytest.mark.parametrize('fault', [None, 'running', 'wrong_step', 'changed_weights'])
def test_reference_checkpoint_is_read_only_and_matches_completed_run(tmp_path, fault):
    run = tmp_path / 'runs/m1/sft-full_20261004_135823_877263'
    checkpoint = run / 'training/checkpoint-1200'
    checkpoint.mkdir(parents=True)
    payloads = {'adapter_config.json': b'{}', 'adapter_model.safetensors': b'weights',
                'trainer_state.json': json.dumps({'global_step': 1200, 'max_steps': 3683}).encode()}
    hashes = {}
    for name, payload in payloads.items():
        (checkpoint / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    (run / 'execution.json').write_text(json.dumps({
        'stage': 'sft-full', 'status': 'running' if fault == 'running' else 'paused',
        'returncode': 0, 'output_checkpoint_integrity': {
            'global_step': 1100 if fault == 'wrong_step' else 1200,
            'scheduler_step': 1200, 'file_sha256': hashes,
        },
    }), encoding='utf-8')
    if fault == 'changed_weights':
        (checkpoint / 'adapter_model.safetensors').write_bytes(b'changed')
    before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    verify = validation_function('verified_checkpoint_1200')['verified_checkpoint_1200']
    if fault:
        with pytest.raises(AssertionError):
            verify(tmp_path)
    else:
        assert verify(tmp_path) == (checkpoint.resolve(), hashes['adapter_model.safetensors'])
    assert before == {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}


def test_completed_validation_is_read_without_restarting_generation(tmp_path):
    adapter, job = tmp_path / 'checkpoint-3683', tmp_path / 'evaluation'
    adapter.mkdir()
    job.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'weights')
    digest = hashlib.sha256(b'weights').hexdigest()
    ids = {str(index): 'image-hash' for index in range(1407)}
    shared = {'versions': {'torch': 'pinned'}, 'annotations_sha256': 'annotations',
              'selected_sha256': 'selection', 'source_files': {'source': 'hash'},
              'image_hashes': ids}
    protocol = {'max_new_tokens': 128}
    manifest = {'inputs': {**shared, 'adapter': str(adapter.resolve()),
                          'adapter_files': {'adapter_model.safetensors': digest}},
                'protocol': protocol}
    report = {'comparable': True, 'canonical_references_checked': True,
              'run_identity_metadata_checked': True, 'n': 1407,
              'expected_split': 'validation', 'issues': [],
              'scores': {'right': {}},
              'explanation_rouge_l_f1': {'comparable': True, 'scores': {'right': .4}}}
    (job / 'full_validation.json').write_text(json.dumps(report), encoding='utf-8')
    (job / 'm1.jsonl.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (job / 'm1.jsonl').write_text(''.join(json.dumps({
        'question_id': qid, 'format_valid': True,
    }) + '\n' for qid in ids), encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    scope = validation_function('evaluate_validation')
    scope.update(cache_manifest={'inputs': shared, 'protocol': protocol},
                 expected_versions=shared['versions'])
    assert scope['evaluate_validation'](adapter, digest, job, 'M1') == report
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    manifest['protocol'] = {'max_new_tokens': 256}
    (job / 'm1.jsonl.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(AssertionError):
        scope['evaluate_validation'](adapter, digest, job, 'M1')


def test_inference_setup_blocks_incomplete_selection_before_source_or_package_changes(tmp_path):
    path = tmp_path / 'reports/m1_selection/epoch_sft-full_20261007_023531_870900/selection.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'running'}), encoding='utf-8')
    source = notebook_cells()['portable-validation-setup']
    with pytest.raises(AssertionError):
        exec(source, {'ROOT': tmp_path})
    assert not (tmp_path / 'source').exists()


def m0_cache_selector():
    source = ast.parse(notebook_cells()['portable-validation-setup'])
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'verified_m0_cache')
    scope = {'hashlib': hashlib, 'Path': Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'cache-selector', 'exec'), scope)
    return scope['verified_m0_cache']


@pytest.mark.historical
@pytest.mark.parametrize('folder', [
    'evaluation_runs/baseline_validation_600_mount_v1',
    'evaluation_local/baseline_validation_800_v1',
    'evaluation_runs/m0_full_validation_cache_v1',
])
def test_m0_cache_selection_accepts_only_the_verified_historical_export(tmp_path, folder):
    root = Path(__file__).resolve().parents[2]
    original = root / 'artifacts/validation_review_20261003/checkpoint_600'
    if not original.is_dir():
        pytest.skip('Historical M0 export is not part of the source release')
    destination = tmp_path / folder
    destination.mkdir(parents=True)
    for name in ('m0.jsonl', 'm0.jsonl.manifest.json'):
        (destination / name).write_bytes((original / name).read_bytes())
    assert m0_cache_selector()(tmp_path) == destination / 'm0.jsonl'


@pytest.mark.historical
def test_m0_cache_restore_is_byte_identical_and_idempotent(tmp_path):
    root = Path(__file__).resolve().parents[2]
    original = root / 'artifacts/validation_review_20261003/checkpoint_600'
    if not original.is_dir():
        pytest.skip('Historical M0 export is not part of the source release')
    uploads, project = tmp_path / 'uploads', tmp_path / 'project'
    uploads.mkdir()
    for name in ('m0.jsonl', 'm0.jsonl.manifest.json'):
        (uploads / name).write_bytes((original / name).read_bytes())
    selector = m0_cache_selector()
    restored = selector(project, uploads)
    assert restored == project / 'evaluation_runs/m0_full_validation_cache_v1/m0.jsonl'
    assert selector(project, uploads) == restored
    for name in ('m0.jsonl', 'm0.jsonl.manifest.json'):
        assert (restored.parent / name).read_bytes() == (original / name).read_bytes()


def test_m0_cache_restore_rejects_wrong_upload_before_creating_output(tmp_path):
    uploads, project = tmp_path / 'uploads', tmp_path / 'project'
    uploads.mkdir()
    for name in ('m0.jsonl', 'm0.jsonl.manifest.json'):
        (uploads / name).write_bytes(b'wrong')
    with pytest.raises(AssertionError):
        m0_cache_selector()(project, uploads)
    assert not project.exists()


def test_m0_cache_selection_does_not_silently_use_changed_or_incomplete_cache(tmp_path):
    selector = m0_cache_selector()
    with pytest.raises(FileNotFoundError, match='No verified M0 cache accessible'):
        selector(tmp_path)
    cache = tmp_path / 'evaluation_runs/baseline_validation_600_mount_v1/m0.jsonl'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'changed\n')
    with pytest.raises(FileNotFoundError):
        selector(tmp_path)
    cache.with_suffix('.jsonl.manifest.json').write_bytes(b'{}')
    with pytest.raises(AssertionError, match='M0 cache changed'):
        selector(tmp_path)


def test_launcher_compiles_and_keeps_definition_separate_from_execution():
    cells = notebook_cells()
    assert {
        'portable-mount', 'portable-source', 'portable-data', 'portable-token-full',
        'portable-m1', 'xGYhAgj9ltcL', 'validation-setup', 'validation-run',
        'portable-evaluation-review',
    } <= cells.keys()
    for name, source in cells.items():
        compile(source, name, "exec")
    history = {name: source for name, source in cells.items()
               if not name.startswith('gpu-gate-')}
    active = {name: source for name, source in cells.items() if name.startswith('gpu-gate-')}
    assert sum(source.count("drive.mount(") for source in history.values()) == 1
    assert sum(source.count("drive.mount(") for source in active.values()) == 1
    source = cells["portable-source"]
    assert "source_20260928_051956_295773" in source
    assert "_verify_source_archive(archive)" in source
    assert "_verify_source_tree(REPO, manifest, manifest_hash)" in source
    with pytest.raises(AssertionError, match='P0'):
        exec(cells["portable-m1"], {'P0_PASSED': False})
    invocation = cells['xGYhAgj9ltcL']
    with pytest.raises(AssertionError, match='P0'):
        exec(invocation, {'P0_PASSED': False})


def test_validation_launcher_uses_separate_source_and_resumable_full_export():
    cells = notebook_cells()
    setup, run = cells['validation-setup'], cells['validation-run']
    assert "source_20261002_062529_455947" in setup
    assert "EVAL_SHA256" in setup and "SOURCE_MANIFEST.json" in setup
    assert "vqa_eval_sources" in setup and "EVAL_ENV['PYTHONPATH']" in setup
    assert "'-e'" not in setup
    assert "verify_record_images" in run
    assert "scripts.run_baseline_validation" in run and "'--resume'" in run
    assert "training_runs/sft-full_20261002_113657_805762/training/checkpoint-800" in run
    assert "['global_step'] == 800" in run
    assert "evaluation_runs/baseline_validation_600_mount_v1" in run
    assert "evaluation_local/baseline_validation_800_v1" in run
    assert "run_m1(" not in run and "sft-full" not in run.split("candidate =")[0]


def test_preference_launcher_generates_unreviewed_calibration_not_training():
    source = notebook_cells()['portable-preference-candidates']
    assert 'SET_AFTER_PACKAGING' not in source
    assert "'scripts.generate_preference_candidates'" in source
    assert "'--count', '100'" in source and "'--count', '2000'" in source
    assert "['global_step'] == 800" in source
    assert "'--resume'" in source
    assert 'scripts.train_dpo' not in source and 'run_m1(' not in source
    assert "'9750ab17cf6dae74680404e167ab1c29d1e2083c8baa2e25bfafcf14bff29c31'" in source


@pytest.fixture
def relocated_validation(tmp_path):
    from scripts.run_baseline_validation import build_commands
    from vqa_explanation.utils.environment import load_environment, project_root

    root = project_root()
    environment = load_environment(root / 'configs/environments/colab_full.example.yaml')
    old, new, checkpoint = (tmp_path / name for name in ('old', 'new', 'checkpoint-600'))
    old.mkdir()
    plan = build_commands(environment, checkpoint.resolve(), old.resolve())
    plan[0][-1] = '/previous/mount/m0.jsonl'
    (old / 'commands.json').write_text(json.dumps(plan), encoding='utf-8')
    predictions = '\n'.join(json.dumps({'question_id': str(i)}) for i in range(1407)) + '\n'
    (old / 'm0.jsonl').write_text(predictions, encoding='utf-8')
    (old / 'm0.jsonl.manifest.json').write_text('{"schema_version": 1}', encoding='utf-8')
    tree = ast.parse(notebook_cells()['validation-run'])
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'prepare_validation_job')
    namespace = {'EVAL_REPO': root, 'json': json, 'hashlib': hashlib,
                 'print': lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'relocation', 'exec'), namespace)
    return namespace['prepare_validation_job'], old, new, checkpoint, environment


def test_validation_relocation_copies_m0_exactly_and_keeps_old_run(relocated_validation):
    prepare, old, new, checkpoint, _ = relocated_validation
    original = {path.name: path.read_bytes() for path in old.iterdir()}
    assert prepare(old, new, checkpoint) == new
    for name in ('m0.jsonl', 'm0.jsonl.manifest.json'):
        assert (new / name).read_bytes() == original[name]
    (new / 'm1.jsonl').write_text('saved M1', encoding='utf-8')
    assert prepare(old, new, checkpoint) == new
    assert (new / 'm1.jsonl').read_text() == 'saved M1'
    assert {path.name: path.read_bytes() for path in old.iterdir()} == original


def test_validation_relocation_does_not_overwrite_conflicts(relocated_validation):
    prepare, old, new, checkpoint, _ = relocated_validation
    prepare(old, new, checkpoint)
    (new / 'm0.jsonl').write_text('conflict', encoding='utf-8')
    with pytest.raises(ValueError, match='M0 differs'):
        prepare(old, new, checkpoint)
    assert (new / 'm0.jsonl').read_text() == 'conflict'
    (new / 'commands.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='different commands'):
        prepare(old, new, checkpoint)


def test_validation_relocation_requires_full_m0(relocated_validation):
    prepare, old, new, checkpoint, _ = relocated_validation
    (old / 'm0.jsonl').write_text('{"question_id": "1"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='complete saved M0'):
        prepare(old, new, checkpoint)
    assert not new.exists()


def test_validation_same_mount_keeps_original_job(relocated_validation):
    from scripts.run_baseline_validation import build_commands

    prepare, old, new, checkpoint, environment = relocated_validation
    plan = build_commands(environment, checkpoint.resolve(), old.resolve())
    (old / 'commands.json').write_text(json.dumps(plan), encoding='utf-8')
    assert prepare(old, new, checkpoint) == old
    assert not new.exists()


def test_build_metadata_restore_does_not_rewrite_source(tmp_path):
    tree = ast.parse(notebook_cells()["portable-source"])
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_restore_build_metadata")
    metadata = tmp_path / "src/package.egg-info/requires.txt"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("generated dependencies", encoding="utf-8")
    source = tmp_path / "src/train.py"
    source.write_text("changed source", encoding="utf-8")
    manifest = {"src/package.egg-info/requires.txt": "unused", "src/train.py": "unused"}
    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/src/package.egg-info/requires.txt", "pinned dependencies")
        archive.writestr("repo/src/train.py", "original source")
    namespace = {"Path": Path, "ARCHIVE_ROOT": "repo/"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "metadata", "exec"), namespace)
    with zipfile.ZipFile(archive_path) as archive:
        namespace["_restore_build_metadata"](tmp_path, archive, manifest)
    assert metadata.read_text(encoding="utf-8") == "pinned dependencies"
    assert source.read_text(encoding="utf-8") == "changed source"
    metadata.unlink()
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="Unsafe build metadata"):
            namespace["_restore_build_metadata"](tmp_path, archive, manifest)


@pytest.mark.parametrize("drift", [False, True])
def test_package_restore_preserves_cuda_build_and_requires_restart(drift):
    source = notebook_cells()["portable-source"]
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_restore_m1_packages")
    installed = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
                 "transformers": "4.57.6", "peft": "0.20.0", "accelerate": "1.14.0",
                 "bitsandbytes": "0.50.2", "pillow": "11.3.0"}
    if drift:
        installed.update(torch="2.11.0+cu130", peft="0.21.0", accelerate="1.15.0")
    calls = []
    namespace = {
        "importlib": SimpleNamespace(metadata=SimpleNamespace(
            version=installed.__getitem__, PackageNotFoundError=KeyError)),
        "subprocess": SimpleNamespace(
            run=lambda command, **kwargs: calls.append((command, kwargs))),
        "sys": SimpleNamespace(executable="python"), "print": lambda *args, **kwargs: None,
    }
    restore_code = compile(ast.Module(body=[function], type_ignores=[]), "package_restore", "exec")
    exec(restore_code, namespace)
    if drift:
        with pytest.raises(RuntimeError, match="Restart"):
            namespace["_restore_m1_packages"]()
        assert len(calls) == 2
        assert "torch==2.11.0+cu128" in calls[0][0]
        assert "torchvision==0.26.0+cu128" in calls[0][0]
        assert calls[0][0][-2:] == ["--index-url", "https://download.pytorch.org/whl/cu128"]
        assert calls[1][0][-2:] == ["peft==0.20.0", "accelerate==1.14.0"]
        assert all(options == {"check": True} for _, options in calls)
    else:
        namespace["_restore_m1_packages"]()
        assert not calls


def test_epoch_launcher_keeps_frozen_trainer_and_checks_before_starting():
    source = notebook_cells()["portable-m1"]
    assert "execution['stage'] == 'sft-full'" in source
    assert "execution['status'] in {'running', 'failed', 'interrupted_or_failed'}" in source
    assert "sft-full_20261006_132346_065189" in source
    assert "checkpoint-3300" in source
    assert "state['global_step'] == 3300" in source
    assert "state['max_steps'] == 3683" in source
    assert "EPOCH_LAUNCHER_SHA" in source
    assert "'--budget-protocol', 'vqa_nle_m1_epoch1_v1'" in source
    assert "'--stop-after-step', '3683'" in source
    assert source.index("['--check-only']") < source.index("assert torch.cuda.is_available()")
    assert source.index("assert torch.cuda.is_available()") < source.index("subprocess.Popen")
    assert "pip" not in source and "train_dpo" not in source
    assert "write_text" not in source
    assert "if execution['status'] == 'running':" in source
    assert "command.append('--recover-interrupted')" in source


def test_checkpoint_audit_uses_pinned_cpu_validator_without_starting_training():
    source = notebook_cells()["portable-checkpoint-audit"]
    assert "c9646355195f940e9cac5a6d63818f6df4c7da591085cea773bc01534ed8256d" in source
    assert source.index("hashlib.sha256(audit_launcher.read_bytes())") < source.index(
        "spec.loader.exec_module(audit_module)"
    )
    assert "verify_checkpoint_payloads(checkpoint)" in source
    assert "state['global_step'] == 1800" in source
    assert "audit_path.open('x'" in source
    assert "Existing audit differs" in source
    assert "subprocess" not in source and "cuda" not in source
    assert "drive.mount(" not in source


def test_checkpoint_audit_rejects_changed_helper_before_importing(tmp_path):
    helper = tmp_path / "source/resume_pinned_m1_c9646355195f.py"
    helper.parent.mkdir()
    helper.write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    before = helper.read_bytes()
    with pytest.raises(AssertionError, match="Audit launcher changed"):
        exec(notebook_cells()["portable-checkpoint-audit"], {"ROOT": tmp_path})
    assert helper.read_bytes() == before
    assert not (tmp_path / "reports").exists()


def test_epoch_audit_is_read_only_for_training_and_rechecks_saved_evidence():
    source = notebook_cells()["portable-epoch-audit"]
    assert "import hashlib" in source and "import json" in source
    assert "execution['status'] == 'passed'" in source
    assert "type(execution['returncode']) is int" in source
    assert source.index("hashlib.sha256(helper.read_bytes())") < source.index(
        "spec.loader.exec_module(checker)"
    )
    assert "verify_epoch_completion(finished_run, 3683)" in source
    assert "verify_checkpoint_payloads(training / 'checkpoint-3683')" in source
    assert "execution['output_checkpoint_integrity'] == integrity" in source
    assert "before_hashes ==" in source
    assert "report_path.open('x'" in source
    assert "if report_is_new:\n    download_epoch_report(report_path)" in source
    assert "subprocess" not in source and "drive.mount(" not in source


def test_epoch_selection_uses_separate_tools_and_original_m1_protocol():
    source = notebook_cells()["portable-epoch-selection"]
    root = Path(__file__).resolve().parents[2]
    for name in ("evaluate_m1_epoch.py", "resume_pinned_m1.py"):
        assert hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() in source
    assert "destination.open('xb')" in source and "selection_is_new" in source
    assert "'--source', str(REPO)" in source
    assert "'configs/environments/colab_full.example.yaml'" in source
    assert "selection['validation_count'] == 256" in source
    assert "selection['test_predictions_generated'] == 0" in source
    assert source.index("hashlib.sha256(payload)") < source.index("subprocess.Popen")
    assert "train_sft" not in source and "drive.mount(" not in source


@pytest.mark.parametrize("status,returncode", [("running", 0), ("passed", 1),
                                              ("passed", False)])
def test_epoch_audit_rejects_incomplete_runs_before_importing(tmp_path, status, returncode):
    run = tmp_path / "runs/m1/sft-full_20261007_023531_870900"
    training = run / "training"
    training.mkdir(parents=True)
    (run / "execution.json").write_text(json.dumps({
        "stage": "sft-full", "status": status, "returncode": returncode,
    }), encoding="utf-8")
    for name in ("training_result.json", "trainer_state.json", "data_plan.json",
                 "run_metadata.json"):
        (training / name).write_text("{}", encoding="utf-8")
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    with pytest.raises(AssertionError):
        exec(notebook_cells()["portable-epoch-audit"], {"ROOT": tmp_path})
    assert before == {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    assert not (tmp_path / "reports").exists()


def test_mount_status_includes_relocated_m1_runs_without_writes(tmp_path):
    run = tmp_path / "runs/m1/sft-full_current"
    checkpoint = run / "training/checkpoint-1300"
    checkpoint.mkdir(parents=True)
    (run / "execution.json").write_text('{"status": "running"}', encoding="utf-8")
    (run / "runner_heartbeat.json").write_text('{"updated_at_utc": "old"}', encoding="utf-8")
    (run / "console.log").write_text("training tail", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1300}', encoding="utf-8")
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    messages = []
    source = notebook_cells()["portable-mount"].split("# Chỉ đọc trạng thái", 1)[1]
    source = source.split("\n", 1)[1]
    exec(source, {"ROOT": tmp_path, "json": json,
                  "print": lambda *args: messages.append(str(args))})
    assert any("sft-full_current" in message for message in messages)
    assert any("checkpoint-1300" in message for message in messages)
    assert any("Heartbeat:" in message for message in messages)
    assert any("training tail" in message for message in messages)
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_token_gate_uses_full_train_and_fixed_validation():
    source = notebook_cells()["portable-token-full"]
    ast.parse(source)
    assert "'--validation-count', '256'" in source
    assert "sft_readiness_full.yaml" in source
    assert "report['summary']['over_max_length'] == 0" in source
    assert "report['summary']['samples'] == 29715" in source
    assert "report['validation_question_ids']" in source
    assert "training_config_sha256" in source
    assert "model_config_sha256" in source
    assert "P0_PASSED = False" in source
    assert source.index("P0_PASSED = False") < source.index("if not TOKEN_REPORT.exists()")
    assert "stdout=subprocess.PIPE" in source
    assert "log.flush()" in source
    assert "if process.wait():" in source


def test_mount_status_reads_previous_reports_without_modifying_them(tmp_path):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    report = run_root / "token_full_M1_v1.json"
    report.write_text('{"summary": {"samples": 29715}}', encoding="utf-8")
    run = run_root / "sft-full_previous"
    checkpoint = run / "training/checkpoint-100"
    checkpoint.mkdir(parents=True)
    (run / "execution.json").write_text('{"status": "paused"}', encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 100}', encoding="utf-8")
    before = {p: p.read_bytes() for p in run_root.rglob("*") if p.is_file()}
    messages = []
    source = notebook_cells()["portable-mount"].split("# Chỉ đọc trạng thái", 1)[1]
    source = source.split("\n", 1)[1]
    exec(source, {"ROOT": tmp_path, "json": json,
                  "print": lambda *args: messages.append(str(args))})
    assert any("29715" in message for message in messages)
    assert any("checkpoint-100" in message for message in messages)
    assert any("chưa có" in message for message in messages)
    assert before == {p: p.read_bytes() for p in run_root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", [None, "wrong_step", "missing_scaler",
                                  "bf16_no_scaler", "invalid_dtype"])
def test_resume_cell_checks_checkpoint_before_loading_launcher(tmp_path, fault):
    checkpoint = tmp_path / "runs/m1/sft-full_20261004_065122_901248/training/checkpoint-1100"
    checkpoint.mkdir(parents=True)
    state = {"global_step": 1000 if fault == "wrong_step" else 1100, "max_steps": 3683}
    (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    dtype = "torch.bfloat16" if fault == "bf16_no_scaler" else "torch.float16"
    if fault == "invalid_dtype":
        dtype = "unknown"
    (checkpoint.parent / "run_metadata.json").write_text(
        json.dumps({"compute_dtype": dtype}), encoding="utf-8"
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt",
                 "adapter_config.json", "adapter_model.safetensors"):
        if fault not in ("missing_scaler", "bf16_no_scaler") or name != "scaler.pt":
            (checkpoint / name).write_bytes(b"fixture")
    before = {file: file.read_bytes() for file in checkpoint.parent.rglob("*") if file.is_file()}
    namespace = {"P0_PASSED": True, "ROOT": tmp_path, "json": json, "print": lambda *args: None}
    source = notebook_cells()["xGYhAgj9ltcL"].split("LAUNCHER_SHA =", 1)[0]
    if fault in ("wrong_step", "missing_scaler", "invalid_dtype"):
        with pytest.raises(AssertionError):
            exec(source, namespace)
    else:
        exec(source, namespace)
    assert before == {file: file.read_bytes() for file in checkpoint.parent.rglob("*")
                      if file.is_file()}
