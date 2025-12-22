import os
import shutil
from pathlib import Path
from data_preprocessing.text_worker import add_info_logging
from models.landmarking_heart import landmarking_computeMeasurements_simplified
from models.implementationGNN import MorphoGCN_Trainer, nnUnet_CandidatePointGenerator


class GNNProject:

    def __init__(self, result_6_nnunet_folder, gnn_folder, train_test_lists):
        self.result_6_nnunet_folder = result_6_nnunet_folder
        self.gnn_folder = gnn_folder
        self.train_test_lists = train_test_lists

    def landmark_nnUnet_generateCandidates(self):
        if os.path.isfile(self.gnn_folder + '/landmark_candidates.json'):
            pass
        else:
            extractor = nnUnet_CandidatePointGenerator(
                json_path = self.result_6_nnunet_folder + '/dataset.json',
                n_candidates = 5,
                min_dist = 1,
                threshold = 0.15,
                include_com = True
            )
            results = extractor.extract_candidate_points(self.result_6_nnunet_folder)
            # save to JSON
            extractor.save_results(results, self.gnn_folder + '/landmark_candidates.json')

    def landmark_GNN_train(self):
        measurment_tester = MorphoGCN_Trainer(landmarking_computeMeasurements_simplified.get_measurement_names())
        measurment_tester.train_morpho_gcn2(self.gnn_folder, self.gnn_folder + '/data/training')

    def landmark_GNN_test(self):
        measurment_tester = MorphoGCN_Trainer(landmarking_computeMeasurements_simplified.get_measurement_names())
        # tester1.test_morpho_gcn_nnUnet(heart_GNN, heart_GNN + '/data/testing', heart_nnUnet + '/Landmarking/temp/landmark_candidates.json')
        measurment_tester.compare_gnn_vs_center(self.gnn_folder, self.gnn_folder + '/data/testing',
                                                self.gnn_folder + '/landmark_candidates.json',
                                                self.gnn_folder + '/results/')

    def configure_folder(self, json_info_folder):
        def _clear_folder(folder):
            """Очищает папку, удаляя все файлы и подпапки"""
            if not folder.exists():
                add_info_logging(f"Folder '{str(folder)}' does not exist.", "work_logger")
                return

            for item in folder.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()  # Удаляем файл или символическую ссылку
                elif item.is_dir():
                    shutil.rmtree(item)  # Удаляем папку рекурсивно

        def _get_file_list(df, series_type, column, suffix, base_path):
            base_path = Path(base_path)
            return [
                base_path / f"{name}{suffix}"
                for name in df[df["type_series"] == series_type][column].dropna()
            ]

        def _copy_img(input_imgs_path, output_folder):
            _clear_folder(output_folder)
            df = self.train_test_lists
            for img_path in input_imgs_path:
                if img_path.name[0] == "H":
                    case_name = df.loc[df["case_name"] == img_path.name[:-5], "used_case_name"].iloc[0]
                    img_path = img_path.with_name(img_path.name.replace("_MJ.json", ".json"))
                    shutil.copy(img_path, output_folder / f"{case_name}.json")
                else:
                    shutil.copy(img_path, output_folder / img_path.name)

        list_train_cases = _get_file_list(self.train_test_lists,
                                         "train",
                                         "case_name",
                                         ".json",
                                         json_info_folder)
        list_test_cases = _get_file_list(self.train_test_lists,
                                        "test"
                                        , "case_name", ".json",
                                        json_info_folder)
        _copy_img(list_train_cases, Path(self.gnn_folder) / "data" / "training")
        _copy_img(list_test_cases, Path(self.gnn_folder) / "data" / "testing")


def process_gnn(result_6_nnunet_folder, gnn_folder, train_test_lists, json_info_folder, create_ds=False,
                training_mod=False, testing_mod=False,):

    gnn_worker = GNNProject(result_6_nnunet_folder=result_6_nnunet_folder,
                            gnn_folder=gnn_folder,
                            train_test_lists=train_test_lists)
    if create_ds:
        gnn_worker.configure_folder(json_info_folder=json_info_folder)
        gnn_worker.landmark_nnUnet_generateCandidates()
    if training_mod:
        gnn_worker.landmark_GNN_train()
    if testing_mod:
        gnn_worker.landmark_GNN_test()
    print('Hi')
