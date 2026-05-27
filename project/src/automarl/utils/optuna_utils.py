
import optuna


def load_study_from_database(database_path: str, study_name='experiment') -> optuna.Study:
    """
    Loads an Optuna study from the given SQLite database file.

    Args:
        database_path (str): The path to the SQLite database file.

    Returns:
        optuna.Study: The loaded Optuna study.
    """
    
    try:
        # Load the study from the storage URI
        study = optuna.load_study(study_name=study_name, storage=database_path)
    
    except Exception as e:
        print(f"Error while trying to load study with name '{study_name}' from database '{database_path}'")
        list_studies_in_database(database_path)
        raise e

    return study

def list_studies_in_database(storage_uri: str) -> list:

    study_list = optuna.get_all_study_summaries(storage=storage_uri)

    print(f"Available studies in storage uri: {storage_uri}")
    for s in study_list:
        print(s.study_name)
