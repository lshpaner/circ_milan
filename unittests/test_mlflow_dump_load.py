import pickle
from sklearn.linear_model import LogisticRegression
from unittest.mock import patch, MagicMock, mock_open
from circ_milan.functions import (
    set_or_create_experiment,
    start_new_run,
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    mlflow_log_parameters_model,
    mlflow_load_model,
    get_run_id_by_name,
)


def test_set_or_create_experiment():
    experiment_name = "test_experiment"
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    with patch(
        "mlflow.get_experiment_by_name",
        return_value=mock_experiment,
    ):
        with patch(
            "mlflow.create_experiment",
            return_value="123",
        ) as mock_create:
            with patch("mlflow.set_experiment"):
                experiment_id = set_or_create_experiment(experiment_name)
                assert experiment_id == "123"
                mock_create.assert_not_called()


def test_start_new_run():
    run_name = "test_run"
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"

    # Mock mlflow.start_run in the first 'with' statement
    with patch("mlflow.start_run", return_value=mock_run):
        # Mock mlflow.end_run in the second 'with' statement
        with patch("mlflow.end_run") as mock_end_run:
            run_id = start_new_run(run_name)

            # Assertions
            assert run_id == "test_run_id"

            # Verify that end_run() was called
            mock_end_run.assert_called_once()


def test_mlflow_dumpArtifact_existing_run():
    experiment_name = "test_experiment"
    run_name = "test_run"
    obj_name = "test_obj"
    obj = {"key": "value"}

    if hasattr(mlflow_dumpArtifact, "artifacts_run_id"):
        del mlflow_dumpArtifact.artifacts_run_id

    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"

    with (
        patch("mlflow.start_run", return_value=mock_run),
        patch("mlflow.log_artifact") as mock_log_artifact,
        patch("os.remove") as mock_remove,
        patch("builtins.open", mock_open(), create=True) as mock_file,
        patch("pickle.dump") as mock_pickle,
        patch(
            "mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="123"),
        ),
        patch("mlflow.set_experiment"),
        patch("mlflow.set_tracking_uri"),
        patch(
            "circ_milan.functions.get_run_id_by_name",
            return_value="test_run_id",
        ),
        patch("circ_milan.functions.start_new_run") as mock_start_new,
    ):
        mlflow_dumpArtifact(
            experiment_name,
            run_name,
            obj_name,
            obj,
            get_existing_id=True,
        )
        mock_file.assert_called_once_with(f"{obj_name}.pkl", "wb")
        mock_pickle.assert_called_once()
        mock_log_artifact.assert_called_once_with(f"{obj_name}.pkl")
        mock_remove.assert_called_once_with(f"{obj_name}.pkl")
        assert mlflow_dumpArtifact.artifacts_run_id == "test_run_id"


def test_mlflow_dumpArtifact_with_artifact_run_id():
    experiment_name = "test_experiment"
    run_name = "test_run"
    obj_name = "test_obj"
    obj = {"key": "value"}
    specific_run_id = "specific_run_id"

    if hasattr(mlflow_dumpArtifact, "artifacts_run_id"):
        del mlflow_dumpArtifact.artifacts_run_id

    mock_run = MagicMock()
    mock_run.info.run_id = specific_run_id

    with (
        patch("mlflow.start_run", return_value=mock_run) as mock_start_run,
        patch("mlflow.log_artifact") as mock_log_artifact,
        patch("os.remove") as mock_remove,
        patch("builtins.open", mock_open(), create=True) as mock_file,
        patch("pickle.dump") as mock_pickle,
        patch(
            "mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="123"),
        ),
        patch("mlflow.set_experiment"),
        patch("mlflow.set_tracking_uri"),
        patch("circ_milan.functions.get_run_id_by_name") as mock_get_run_id,
        patch("circ_milan.functions.start_new_run") as mock_start_new,
    ):
        # Set the attribute before the call to mimic the function's behavior
        mlflow_dumpArtifact.artifacts_run_id = specific_run_id

        mlflow_dumpArtifact(
            experiment_name,
            run_name,
            obj_name,
            obj,
            get_existing_id=False,
            artifact_run_id=specific_run_id,
        )

        mock_file.assert_called_once_with(f"{obj_name}.pkl", "wb")
        mock_pickle.assert_called_once()
        mock_log_artifact.assert_called_once_with(f"{obj_name}.pkl")
        mock_remove.assert_called_once_with(f"{obj_name}.pkl")
        assert mlflow_dumpArtifact.artifacts_run_id == specific_run_id
        mock_get_run_id.assert_not_called()
        mock_start_new.assert_not_called()
        mock_start_run.assert_called_once_with(
            run_id=specific_run_id,
            nested=True,
        )


def test_mlflow_loadArtifact():
    experiment_name = "test_experiment"
    run_name = "test_run"
    obj_name = "test_obj"
    obj = {"key": "value"}
    local_path = f"{obj_name}.pkl"

    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    with (
        patch(
            "mlflow.tracking.MlflowClient.download_artifacts",
            return_value=local_path,
        ),
        patch(
            "builtins.open",
            mock_open(read_data=pickle.dumps(obj)),
            create=True,
        ),
        patch("pickle.load", return_value=obj),
        patch(
            "mlflow.tracking.MlflowClient.search_runs",
            return_value=[mock_run],
        ),
        patch(
            "mlflow.get_experiment_by_name",
            return_value=mock_experiment,
        ),
        patch("mlflow.set_experiment"),
    ):
        loaded_obj = mlflow_loadArtifact(
            experiment_name,
            run_name,
            obj_name,
        )
        assert loaded_obj == obj


def test_mlflow_log_parameters_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    model_type = "lr"
    n_iter = 10
    kfold = True
    outcome = "target"
    model_name = "test_model"
    model = LogisticRegression()
    hyperparam_dict = {
        "param1": 1,  # Scalar value instead of [1, 2]
        "param2": "value",
    }

    with (
        patch("mlflow.start_run"),
        patch("mlflow.log_param"),
        patch("mlflow.sklearn.log_model"),
        patch(
            "mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="123"),
        ),
        patch("mlflow.set_experiment"),
    ):
        mlflow_log_parameters_model(
            model_type,
            n_iter,
            kfold,
            outcome,
            run_name,
            experiment_name,
            model_name,
            model,
            hyperparam_dict,
        )


def test_mlflow_load_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    model_name = "test_model"
    mock_model = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_run.info.artifact_uri = "file:///tmp/test_run_id/artifacts"

    with (
        patch(
            "mlflow.tracking.MlflowClient.search_runs",
            return_value=[mock_run],
        ),
        patch(
            "mlflow.tracking.artifact_utils._download_artifact_from_uri",
            return_value="/tmp/fake_model_path",
        ) as mock_download,
        patch(
            "mlflow.sklearn.load_model",
            return_value=mock_model,
        ) as mock_load,
        patch(
            "mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="123"),
        ),
        patch("mlflow.set_experiment"),
    ):
        print("Calling mlflow_load_model...")
        loaded_model = mlflow_load_model(
            experiment_name,
            run_name,
            model_name,
        )
        print(f"Loaded model: {loaded_model}")
        assert loaded_model == mock_model
        mock_load.assert_called_once_with(f"runs:/test_run_id/{model_name}")
        print(f"_download_artifact_from_uri called: {mock_download.called}")


def test_get_run_id_by_name_existing():
    experiment_name = "test_experiment"
    run_name = "test_run"
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"

    # Patch at the module level where the function is defined
    with (
        patch(
            "circ_milan.functions.MlflowClient",
            autospec=True,
        ) as mock_client_class,
        patch(
            "circ_milan.functions.mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="123"),
        ) as mock_get_exp,
        patch("circ_milan.functions.start_new_run") as mock_start_new,
    ):
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.search_runs.return_value = [mock_run]

        run_id = get_run_id_by_name(experiment_name, run_name)
        assert run_id == "test_run_id"
        mock_client_instance.search_runs.assert_called_once_with(
            experiment_ids=["123"],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
        )
        mock_start_new.assert_not_called()
        mock_get_exp.assert_called_once_with(experiment_name)


def test_get_run_id_by_name_create_new():
    experiment_name = "test_experiment"
    run_name = "test_run"
    new_run_id = "new_run_id"

    # Patch at the module level where the function is defined
    with (
        patch(
            "circ_milan.functions.MlflowClient",
            autospec=True,
        ) as mock_client_class,
        patch(
            "circ_milan.functions.mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="123"),
        ) as mock_get_exp,
        patch(
            "circ_milan.functions.start_new_run", return_value=new_run_id
        ) as mock_start_new,
    ):
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.search_runs.return_value = []

        run_id = get_run_id_by_name(experiment_name, run_name)
        assert run_id == new_run_id
        mock_client_instance.search_runs.assert_called_once_with(
            experiment_ids=["123"],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
        )
        mock_start_new.assert_called_once_with(run_name)
        mock_get_exp.assert_called_once_with(experiment_name)
