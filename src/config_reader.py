import yaml
import os

class Config:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, "configs", "config.yaml")
        with open(config_path, 'r') as f:
            self._cfg = yaml.safe_load(f)

    def get(self, *keys):
        val = self._cfg
        for k in keys:
            val = val[k]
        return val
    
    def is_feature_enabled(self, feature_name):
        return self._cfg['features'].get(feature_name, False)