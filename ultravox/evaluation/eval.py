import re

from ultravox.evaluation import eval_types
from ultravox.evaluation import gpt_eval
from ultravox.evaluation import wer


def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric == "asr":
        return wer.evaluate_answer_asr(sample)
    elif metric == "boolq":
        return gpt_eval.evaluate_answer_boolq(sample)
    elif metric == "instruct":
        return gpt_eval.evaluate_answer_instruct(sample)
    elif metric == "binary":
        last_words = re.findall(r"\b\w+\b(?=\W*$)", sample.generated_answer)
        if not last_words:
            return eval_types.InstructResult(score=0, reason="No last word found")
        last_word: str = last_words[-1].lower()
        if last_word in ["yes", "true"]:
            last_word = "true"
        elif last_word in ["no", "false"]:
            last_word = "false"
        else:
            return eval_types.InstructResult(score=0, reason="Last word not true/false")
        return eval_types.InstructResult(
            score=last_word == sample.expected_answer.lower(),
            reason="exact_match check",
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")
