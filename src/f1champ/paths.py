from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw" / "jolpica"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions" / "v1"

    @property
    def validation(self) -> Path:
        return self.root / "validation"

    def ensure(self) -> None:
        for path in (self.raw, self.processed, self.features, self.predictions, self.validation):
            path.mkdir(parents=True, exist_ok=True)


def resolve_data_dir(path: str | Path) -> DataPaths:
    return DataPaths(Path(path).resolve())
