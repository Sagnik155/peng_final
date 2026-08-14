import json
from pathlib import Path

def parse_clicks(json_path: Path) -> list[dict]:
    """
    Parses the PENGWIN click JSON.
    Returns a list of dictionaries containing the coordinate and anatomy name.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Click file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    parsed_clicks = []
    
    for point_data in data.get("points", []):
        name = point_data.get("name", "")
        coord = point_data.get("point", [])
        
        if not coord or len(coord) != 3:
            continue
            
        anatomy_target = "background"
        name_lower = name.lower()
        if "femur" in name_lower:
            anatomy_target = "femur"
        elif "left hipbone" in name_lower:
            anatomy_target = "left_hipbone"
        elif "right hipbone" in name_lower:
            anatomy_target = "right_hipbone"
        elif "sacrum" in name_lower:
            anatomy_target = "sacrum"

        parsed_clicks.append({
            "name": name,
            "anatomy": anatomy_target,
            # FIXED: PENGWIN natively stores JSON points in [z, y, x] order
            "z": int(coord[0]),
            "y": int(coord[1]),
            "x": int(coord[2])
        })
        
    return parsed_clicks