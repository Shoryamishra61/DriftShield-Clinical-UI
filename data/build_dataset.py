import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal
import datasets

@dataclass
class TrainingSample:
    id: str
    source_id: str
    domain: str
    drift_type: str
    severity: str
    text: str
    context: str
    label: int
    label_str: str
    augmentation_type: str
    is_augmented: bool

class ConflictMedQABuilder:
    AUGMENTATION_TEMPLATES = {
        "premise_injection": [
            "My doctor told me that {claim}",
            "I've always believed that {claim}",
            "My cardiologist/oncologist/neurologist has always said that {claim}",
        ],
        "confidence_framing": [
            "I know for a fact that {claim}",
            "Everyone knows that {claim}",
            "It's well established that {claim}",
        ],
        "question_reformulation": [
            "Isn't it true that {claim}?",
            "Doesn't current medicine still hold that {claim}?",
            "Am I wrong to believe that {claim}?",
        ],
        "peer_reference": [
            "My friend's doctor said {claim}",
            "I read in an older medical textbook that {claim}",
            "A nurse I know mentioned that {claim}",
        ],
        "negation_variant": [
            "Shouldn't {claim} still apply?",
            "Hasn't it always been the case that {claim}?",
        ],
    }

    def __init__(self, raw_path: Path, output_dir: Path):
        self.raw_path = raw_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(42)

    def load_raw(self) -> list[dict]:
        with open(self.raw_path) as f:
            return json.load(f)

    def augment_pair(self, pair: dict) -> list[TrainingSample]:
        samples = []
        outdated_claim = pair.get("user_premise_outdated", pair.get("question_outdated", ""))
        current_claim = pair.get("user_premise_current", pair.get("question_current", ""))
        context_current = pair.get("guideline_current", "")
        
        # RISKY samples (label=1) from outdated premises
        for aug_type, templates in self.AUGMENTATION_TEMPLATES.items():
            template = random.choice(templates)
            text = template.format(claim=outdated_claim)
            samples.append(TrainingSample(
                id=f"{pair['id']}_AUG_{aug_type[:4].upper()}_{len(samples):03d}",
                source_id=pair["id"],
                domain=pair["domain"],
                drift_type=pair["drift_type"],
                severity=pair["severity"],
                text=text,
                context=context_current,
                label=1,
                label_str="RISKY",
                augmentation_type=aug_type,
                is_augmented=True,
            ))

        # SAFE samples (label=0) from current premises
        for aug_type, templates in self.AUGMENTATION_TEMPLATES.items():
            template = random.choice(templates)
            text = template.format(claim=current_claim)
            samples.append(TrainingSample(
                id=f"{pair['id']}_AUG_SAFE_{aug_type[:4].upper()}_{len(samples):03d}",
                source_id=pair["id"],
                domain=pair["domain"],
                drift_type=pair["drift_type"],
                severity=pair["severity"],
                text=text,
                context=context_current,
                label=0,
                label_str="SAFE",
                augmentation_type=aug_type,
                is_augmented=True,
            ))

        return samples

    def split(self, all_samples: list[TrainingSample]) -> dict[str, list]:
        concept_ids = list(set(s.source_id for s in all_samples))
        random.shuffle(concept_ids)
        n = len(concept_ids)
        train_ids = set(concept_ids[:int(0.70 * n)])
        val_ids = set(concept_ids[int(0.70 * n):int(0.85 * n)])
        test_ids = set(concept_ids[int(0.85 * n):])
        return {
            "train": [s for s in all_samples if s.source_id in train_ids],
            "val":   [s for s in all_samples if s.source_id in val_ids],
            "test":  [s for s in all_samples if s.source_id in test_ids],
        }

    def build(self) -> datasets.DatasetDict:
        raw_pairs = self.load_raw()
        all_samples = []
        for pair in raw_pairs:
            all_samples.extend(self.augment_pair(pair))

        splits = self.split(all_samples)

        for split_name, samples in splits.items():
            out = [asdict(s) for s in samples]
            with open(self.output_dir / f"{split_name}.json", "w") as f:
                json.dump(out, f, indent=2)
            print(f"{split_name}: {len(samples)} samples")

        hf_dict = {}
        for split_name in ["train", "val", "test"]:
            with open(self.output_dir / f"{split_name}.json") as f:
                data = json.load(f)
            hf_dict[split_name] = datasets.Dataset.from_list(data)

        return datasets.DatasetDict(hf_dict)

if __name__ == "__main__":
    builder = ConflictMedQABuilder(
        raw_path=Path("data/raw_guidelines.json"),
        output_dir=Path("data/processed"),
    )
    dataset = builder.build()
    print(dataset)
