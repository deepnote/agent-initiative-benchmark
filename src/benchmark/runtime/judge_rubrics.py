"""The benchmark's 0 to 10 scoring rubric."""

RUBRIC_VERSION = "benchmark-rubric-v1"
SCORE_MAX = 10


def score_max(version):
    if version != RUBRIC_VERSION:
        raise ValueError("Unknown scoring rubric")
    return SCORE_MAX


def uses_content_quality(version):
    return version == RUBRIC_VERSION


def criterion_labels(version):
    score_max(version)
    return {
        "intervention_or_silence": "Decision to help",
        "correctness_and_evidence": "Content quality",
        "manner_of_intervention": "Clarity and tone",
        "audience_and_location": "Audience and place",
    }
