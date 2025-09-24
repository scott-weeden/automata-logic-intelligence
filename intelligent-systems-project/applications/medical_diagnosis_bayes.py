"""Naive Bayes medical diagnosis example."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, exp
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class DiseaseModel:
    name: str
    prior: float
    symptom_likelihood: Dict[str, float]


class BayesDiagnosis:
    def __init__(self, diseases: Iterable[DiseaseModel]) -> None:
        self.diseases = list(diseases)
        if not self.diseases:
            raise ValueError("At least one disease model is required")

    def infer(self, observed_symptoms: Iterable[str]) -> List[Tuple[str, float]]:
        symptoms = list(observed_symptoms)
        log_posteriors: List[Tuple[str, float]] = []

        for disease in self.diseases:
            log_prob = log(disease.prior)
            for symptom in symptoms:
                likelihood = disease.symptom_likelihood.get(symptom, 0.05)
                log_prob += log(max(likelihood, 1e-6))
            log_posteriors.append((disease.name, log_prob))

        normaliser = self._log_sum_exp([lp for _, lp in log_posteriors])
        return [(name, exp(lp - normaliser)) for name, lp in log_posteriors]

    @staticmethod
    def _log_sum_exp(values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            return float('-inf')
        m = max(values)
        return m + log(sum(exp(v - m) for v in values))


def _build_default_model() -> BayesDiagnosis:
    diseases = [
        DiseaseModel(
            name="Common Cold",
            prior=0.15,
            symptom_likelihood={
                "cough": 0.7,
                "sore_throat": 0.6,
                "fatigue": 0.5,
                "runny_nose": 0.8,
            },
        ),
        DiseaseModel(
            name="Influenza",
            prior=0.08,
            symptom_likelihood={
                "cough": 0.85,
                "fever": 0.95,
                "fatigue": 0.9,
                "body_aches": 0.9,
            },
        ),
        DiseaseModel(
            name="COVID-19",
            prior=0.05,
            symptom_likelihood={
                "cough": 0.7,
                "fever": 0.8,
                "loss_of_taste": 0.6,
                "fatigue": 0.75,
            },
        ),
    ]
    return BayesDiagnosis(diseases)


def diagnose_patient(symptoms: Iterable[str], show_output: bool = True) -> List[Tuple[str, float]]:
    model = _build_default_model()
    ranking = model.infer(symptoms)
    ranking.sort(key=lambda item: item[1], reverse=True)

    if show_output:
        print("Observed symptoms:", ", ".join(symptoms))
        print("Posterior probabilities:")
        for name, prob in ranking:
            print(f"  {name:12} -> {prob:0.3f}")
    return ranking


if __name__ == "__main__":  # pragma: no cover - manual demo
    diagnose_patient(["cough", "fever", "fatigue"], show_output=True)
