
import yaml
from pydantic import BaseModel
from typing import Any


class ProjectConfig(BaseModel):
    num_features: list[str]
    cat_features: list[str]
    target_feature: str
    catalog_name: str
    schema_name: str
    dataset_name: str
    parameters: dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """ :param config_path: Path to the YAML config file
            :param env: Environment name to load environment-specific settings
            :return: ProjectConfig instance initialized with parsed config"""
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected: 'prd', 'acc' or 'dev'")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"]= config_dict[env]["schema_name"]

            return cls(**config_dict)
        
class Tags(BaseModel):
    git_sha: str
    branch: str
    job_run_id: str