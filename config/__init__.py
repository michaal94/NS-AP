import json

DEFAULT_CONFIG = {
    "segmentation": "config/segmentation_default.json",
    "instructions": "config/instructions_default.json",
    "attributes": "config/attributes_default.json",
    "actions": "config/action_planning_default.json",
    "attributes_ycb": "config/attributes_default_ycb.json"
}

def get_config(stage):
    assert stage in DEFAULT_CONFIG
    with open(DEFAULT_CONFIG[stage], 'r') as f:
        return json.load(f)