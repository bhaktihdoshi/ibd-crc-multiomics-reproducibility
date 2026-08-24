from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_project_root() -> Path:
    # This file lives at <PROJECT>/thesis_preprocessing_workflow/stage3_ibd_crc_integration/scripts/.
    return Path(__file__).resolve().parents[3]


_PROJECT_ROOT = Path(os.environ["THESIS_PROJECT_ROOT"]) if os.environ.get("THESIS_PROJECT_ROOT") else _default_project_root()


@dataclass(frozen=True)
class Stage3Config:
    root: Path = _PROJECT_ROOT
    random_seed: int = 42

    @property
    def stage3_dir(self) -> Path:
        return self.root / "thesis_preprocessing_workflow" / "stage3_ibd_crc_integration"

    @property
    def ibd_dir(self) -> Path:
        return self.root / "thesis_preprocessing_workflow" / "ibd_pipeline"

    @property
    def crc_stage2_dir(self) -> Path:
        return self.root / "thesis_preprocessing_workflow" / "crc_pipeline" / "stage2"

    @property
    def required_dirs(self) -> list[Path]:
        return [
            self.stage3_dir / "config",
            self.stage3_dir / "scripts",
            self.stage3_dir / "tables",
            self.stage3_dir / "figures",
            self.stage3_dir / "reports",
            self.stage3_dir / "manifests",
            self.stage3_dir / "logs",
            self.stage3_dir / "tests",
        ]


CONFIG = Stage3Config()

