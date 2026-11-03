"""Build the WeavEHR project read by the `weavehr` backend: WeavEHR's extraction and concept steps for one
dataset, written to `weavehr_project_path` in config.yaml. Requires WeavEHR (Python >= 3.13).
"""
import argparse
import tempfile
from pathlib import Path

import yaml
from weavehr import ConceptStep, ExtractionStep, WeavEHRProject
from weavehr.config.registry import load_configs
from weavehr.logging import configure_logging
from weavehr.steps.concept.config.concept import ConceptConfig
from weavehr.steps.concept.registry import concept_config_registry
from weavehr.steps.extraction.registry import dataset_config_registry
from weavehr.utils.loader import auto_load_configs, bundled_config_path

from src import conf


def register_extensions(extensions: Path) -> None:
    """Register the YAIB additions to WeavEHR's configs ahead of WeavEHR's own: the concepts and table configs in
    weavehr-extensions, and its concept mappings, which replace WeavEHR's for that dataset."""
    bundled = bundled_config_path()
    versions = [p for p in sorted((extensions / 'datasets').glob('*/*')) if p.is_dir()]
    for version in versions:
        if (version / 'tables').is_dir() or (version / 'extends.yml').is_file():
            dataset_config_registry.load(version / 'tables')
    mappings = [p / 'mappings' for p in versions + sorted((bundled / 'datasets').glob('*/*')) if p.is_dir()]
    for root in (extensions, bundled):
        for concept in load_configs(root / 'concepts', config_type=ConceptConfig, dataset_paths=mappings):
            concept_config_registry.register(concept)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='name of the WeavEHR dataset, e.g., mimic-iv-demo')
    parser.add_argument('--version', required=True, help='version of the dataset, e.g., 2.2')
    parser.add_argument('--data-path', required=True, help='directory with the raw data files of the dataset')
    parser.add_argument('--overwrite', action='store_true', help='rebuild steps that have already been run')
    args = parser.parse_args()

    dataset = {'name': args.src, 'version': args.version}
    # the source event and code of concept rows, to look up the ICU stay they were recorded for
    source = {'extension_columns': {'event': 'col("event")', 'source_code': 'col("code")'}}
    steps = {
        'extraction': {'name': 'Extraction', 'version': '1.0.0', 'overwrite': args.overwrite,
                       'config': {'data': [{**dataset, 'path': str(Path(args.data_path).resolve())}]}},
        'concept': {'name': 'Concept', 'version': '1.0.0', 'overwrite': args.overwrite,
                    'config': {'extraction_step': 'Extraction', 'mapping_configs': [{**dataset, **source}]}},
    }

    configure_logging(level="INFO")
    register_extensions(Path('../weavehr-extensions'))
    auto_load_configs()
    project_path = Path(conf['weavehr_project_path'].format(src=args.src))
    with tempfile.TemporaryDirectory() as config_dir, WeavEHRProject(project_path) as project:
        for name, step in steps.items():
            with open(Path(config_dir) / f'{name}.yml', 'w') as stream:
                yaml.safe_dump(step, stream)
        ExtractionStep.load(project, Path(config_dir) / 'extraction.yml').run()
        ConceptStep.load(project, Path(config_dir) / 'concept.yml').run()
    print(f'WeavEHR project for {args.src} written to {project_path}')
