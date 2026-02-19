"""Custom LightEval tasks aligned with SmolLM3 evals."""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation
from lighteval.utils.language import Language

QA_METRICS = [
    LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]
ALL_QA_FORMULATIONS = [MCFFormulation(), CFFormulation(), HybridFormulation()]

# fmt: off
MMLU_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
]
# fmt: on

HELLASWAG_TASKS = [
    LightevalTaskConfig(
        name=f"hellaswag_{formulation.name.lower()}",
        prompt_function=get_hellaswag_prompt_function(
            language=Language.ENGLISH,
            adapter=lambda line: {
                "activity_label": line["activity_label"],
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_avail_splits=("train", "validation"),
        evaluation_splits=("validation",),
        metrics=get_metrics_for_formulation(formulation, QA_METRICS),
    )
    for formulation in ALL_QA_FORMULATIONS
]

MMLU_TASKS = [
    LightevalTaskConfig(
        name=f"mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="cais/mmlu",
        hf_subset=subset,
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        hf_avail_splits=("auxiliary_train", "dev", "validation", "test"),
        evaluation_splits=("test",),
        few_shots_split="dev",
        metrics=get_metrics_for_formulation(formulation, QA_METRICS),
    )
    for subset in MMLU_SUBSETS
    for formulation in ALL_QA_FORMULATIONS
]

MMLU_PRO_METRICS = [
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]
MMLU_PRO_FORMULATIONS = [CFFormulation(), MCFFormulation()]

MMLU_PRO_TASKS = [
    LightevalTaskConfig(
        name=f"mmlu_pro_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metrics=get_metrics_for_formulation(formulation, MMLU_PRO_METRICS),
    )
    for formulation in MMLU_PRO_FORMULATIONS
]

MMLU_PRO_BIOLOGY_TASKS = [
    LightevalTaskConfig(
        name=f"mmlu_pro_biology_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        hf_repo="hf-carbon/mmlu-pro-biology",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metrics=get_metrics_for_formulation(formulation, MMLU_PRO_METRICS),
    )
    for formulation in MMLU_PRO_FORMULATIONS
]

TASKS_TABLE = HELLASWAG_TASKS + MMLU_TASKS + MMLU_PRO_TASKS + MMLU_PRO_BIOLOGY_TASKS
