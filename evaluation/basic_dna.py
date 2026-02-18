"""Custom Basic DNA CF task aligned with SmolLM3 evals."""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language

qa_metrics = [
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]
formulation = CFFormulation()

TASKS_TABLE = [
    LightevalTaskConfig(
        name="basic_dna_cf",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        hf_repo="hf-carbon/basic-dna",
        hf_subset="default",
        evaluation_splits=("train",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
]
