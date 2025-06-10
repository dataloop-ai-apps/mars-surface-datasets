import os
import json
import time
import urllib
import logging
import pathlib
import zipfile
import dtlpy as dl

from tqdm import tqdm
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor

urllib.request.urlcleanup()

logger = logging.getLogger("[dataset-loader]")


class DatasetLoader(dl.BaseServiceRunner):
    def __init__(self):
        self.tmp_path = None

    def load_unannotated(self, dataset: dl.Dataset, source: str, progress: dl.Progress = None):
        if self.tmp_path is None:
            self.tmp_path = os.path.join(os.getcwd(), 'tmp')
            if not os.path.exists(self.tmp_path):
                os.makedirs(self.tmp_path)

        if progress:
            progress.update(message="Preparing data")
        # Downloading
        tmp_zip_path = os.path.join(self.tmp_path, 'data.zip')
        urlretrieve(source, tmp_zip_path)
        # Unzip
        data_dir = os.path.join(self.tmp_path, 'data')
        zip_ref = zipfile.ZipFile(tmp_zip_path, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Upload dataset with features
        if progress:
            progress.update(message="Uploading dataset with features")
        self.upload_dataset(dataset=dataset, data_path=data_dir, progress=progress)

    @staticmethod
    def upload_dataset(data_path, dataset: dl.Dataset, progress: dl.Progress):
        ontology_json_folder_path = os.path.join(data_path, 'ontology')
        items_folder_path = os.path.join(data_path, 'items')
        annotation_jsons_folder_path = os.path.join(data_path, 'json')

        # Upload ontology if exists
        if os.path.exists(ontology_json_folder_path) is True:
            ontology_json_filepath = list(pathlib.Path(ontology_json_folder_path).rglob('*.json'))[0]
            with open(ontology_json_filepath, 'r') as f:
                ontology_json = json.load(f)
            ontology: dl.Ontology = dataset.ontologies.list()[0]
            ontology.copy_from(ontology_json=ontology_json)
        item_binaries = sorted(list(filter(lambda x: x.is_file(), pathlib.Path(items_folder_path).rglob('*'))))
        annotation_jsons = sorted(list(pathlib.Path(annotation_jsons_folder_path).rglob('*.json')))

        try:
            for attempt in range(3):
                try:
                    feature_set = dataset.project.feature_sets.create(
                        name="openai-clip", size=512, entity_type="item", set_type="clip"
                    )
                    logger.info("Creating new feature set for openai-clip")
                    break
                except dl.exceptions.BadRequest:  # When feature set already exists
                    feature_set = dataset.project.feature_sets.get(feature_set_name="openai-clip")
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        logger.error(f"Failed to create or get feature set after 3 attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                    time.sleep(1)  # Wait before retrying
        except Exception as e:
            raise ValueError(f"Failed to create or get feature set after 3 attempts: {str(e)}")

        def process_item(args):
            item_file, annotation_file, dataset, feature_set = args
            # Load annotation json
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)

            # Extract description, tags and user metadata
            item_metadata = dict()
            item_metadata["user"] = annotation_data.get("metadata", dict()).get("user", dict())
            tags_metadata = annotation_data.get("metadata", dict()).get("system", dict()).get('tags', None)
            if tags_metadata is not None:
                item_metadata.update({"system": {"tags": tags_metadata}})
            caption = annotation_data.get("description", None)

            # Get features
            features = annotation_data.get('itemVectors', [])
            feature = features[0]

            # Construct item remote path
            remote_path = f"/{item_file.parent.stem}"
            item = dataset.items.upload(local_path=str(item_file), remote_path=remote_path, item_metadata=item_metadata, item_description=caption)
            item.features.create(
                value=feature["value"], project_id=dataset.project.id, feature_set_id=feature_set.id, entity=item
            )

        # Create arguments list for thread pool
        args_list = [
            (item_file, annotation_file, dataset, feature_set)
            for item_file, annotation_file in zip(item_binaries, annotation_jsons)
        ]

        # Process items using thread pool with progress bar
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_item, args_list), total=len(args_list), desc="Uploading items and features"))
        return dataset


if __name__ == "__main__":
    project = dl.projects.create(project_name=f"TEST mars surface {time.time()}")
    test_dataset = project.datasets.create(dataset_name=f"TEST mars surface {time.time()}")
    DatasetLoader().load_unannotated(
        progress=dl.Progress(),
        dataset=test_dataset,
        source="https://storage.googleapis.com/model-mgmt-snapshots/datasets_mars_surface_captioning/mars_surface_captioned.zip",
    )
    test_dataset.delete(sure=True, really=True)
    project.delete(sure=True, really=True)
