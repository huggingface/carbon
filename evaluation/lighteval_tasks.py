"""Custom LightEval tasks aligned with SmolLM3 evals."""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation
from lighteval.utils.language import Language

qa_metrics = [
    LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]
all_qa_formulations = [MCFFormulation(), CFFormulation()]

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
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
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
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subset in MMLU_SUBSETS
    for formulation in all_qa_formulations
]

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
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
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
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]

LAB_BENCH_SEQQA_SUBTASKS = [
    "ORF-seq-AAid-v1-public",
    "ORF-seq-AAseq-v1-public",
    "ORF-seq-numlen-v1-public",
    "ORF-transeff-v1-public",
    "PCR-gene-enzprimers-v1-public",
    "PCR-gene-gibshindprimers-v1-public",
    "PCR-gene-gibssmaprimers-v1-public",
    "PCR-geneprimers-enz-v1-public",
    "PCR-len-primers-v1-public",
    "PCR-primers-len-v1-public",
    "PCR-seq-enzprimers-v1-public",
    "PCR-seq-primers-v1-public",
    "Prop-seq-gcpercent-v1-public",
    "RE-seq-lenfrags-v1-public",
    "RE-seq-numfrags-v1-public",
]

LAB_BENCH_SEQQA_TASKS = [
    LightevalTaskConfig(
        name=f"lab_bench_seqqa_{formulation.name.lower()}:{subtask}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": int(line["answer_index"]),
            },
            formulation=formulation,
        ),
        hf_repo="hf-carbon/lab-bench",
        hf_subset="SeqQA",
        hf_filter=(None if subtask == "all" else lambda line, subtask=subtask: line["subtask"] == subtask),
        hf_avail_splits=("train",),
        evaluation_splits=("train",),
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for subtask in ["all"] + LAB_BENCH_SEQQA_SUBTASKS
    for formulation in all_qa_formulations
]

BASIC_DNA_TASKS = [
    LightevalTaskConfig(
        name=f"basic_dna_{formulation.name.lower()}",
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
    for formulation in all_qa_formulations
]

GPQA_BIOLOGY_MCQ_TASKS = [
    LightevalTaskConfig(
        name=f"gpqa_biology_mcq_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        hf_repo="hf-carbon/gpqa-biology-mcq",
        hf_subset="gpqa_main",
        evaluation_splits=("train",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]

SCIEVAL_MCQ_TASKS = [
    LightevalTaskConfig(
        name=f"scieval_mcq_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        hf_repo="hf-carbon/scieval-biology",
        hf_subset="mcq",
        evaluation_splits=("test_mcq",),
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]

SCIEVAL_MCQ_GENETICS_TASKS = [
    LightevalTaskConfig(
        name=f"scieval_mcq_genetics_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        hf_repo="hf-carbon/scieval-biology",
        hf_subset="mcq_genetics",
        evaluation_splits=("test_mcq",),
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]

SCIKNOWEVAL_MCQ_TASKS = [
    LightevalTaskConfig(
        name=f"sciknoweval_mcq_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        hf_repo="hf-carbon/sciknoweval-biology",
        hf_subset="mcq-4-choices-formatted",
        evaluation_splits=("test",),
        metrics=get_metrics_for_formulation(formulation, qa_metrics),
    )
    for formulation in all_qa_formulations
]

TASKS_TABLE = (
    HELLASWAG_TASKS
    + MMLU_TASKS
    + MMLU_PRO_TASKS
    + MMLU_PRO_BIOLOGY_TASKS
    + LAB_BENCH_SEQQA_TASKS
    + BASIC_DNA_TASKS
    + GPQA_BIOLOGY_MCQ_TASKS
    + SCIEVAL_MCQ_TASKS
    + SCIEVAL_MCQ_GENETICS_TASKS
    + SCIKNOWEVAL_MCQ_TASKS
)
