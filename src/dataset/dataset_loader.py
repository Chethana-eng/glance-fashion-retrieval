from pathlib import Path
from typing import Dict, List

class DatasetLoader:
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
    
    def load_images(self) -> List[Dict]:
        image_metadata = []
        suported_extensions = {'.jpg', '.jpeg', '.png'} #optimization using set instead of list
        image_files = [
            file
            for file in self.image_dir.rglob("*")
            if file.suffix.lower() in suported_extensions
        ]
        for idx, image_path in enumerate(image_files):
            image_metadata.append({
                "id": idx,
                "path": str(image_path),
                "filename": image_path.name
            })
        print(f"Loaded {len(image_metadata)} images.")
        return image_metadata