import math


def least_confident(conf, k):
    """Indices of the k rows with the lowest confidence (stable for ties)."""
    return sorted(range(len(conf)), key=lambda i: conf[i])[:k]


def escalation_curve(conf, student_correct, llm_correct, ks):
    """Exact router accuracy if the k least confident rows are sent to the LLM.

    conf: student confidence per row. student_correct: bool per row.
    llm_correct: {row index: bool}, must contain the least confident rows.
    """
    n = len(conf)
    order = sorted(range(n), key=lambda i: conf[i])
    total_ok = sum(student_correct)
    out = []
    for k in ks:
        escalated = order[:k]
        missing = [i for i in escalated if i not in llm_correct]
        if missing:
            raise KeyError(f"no LLM answer for {len(missing)} of the {k} least confident rows")
        student_ok = sum(student_correct[i] for i in escalated)
        llm_ok = sum(llm_correct[i] for i in escalated)
        out.append({
            "k": k,
            "escalated": k / n,
            "threshold": conf[order[k - 1]] if k else None,
            "router_acc": (total_ok - student_ok + llm_ok) / n,
            "student_acc_on_escalated": student_ok / k if k else math.nan,
            "llm_acc_on_escalated": llm_ok / k if k else math.nan,
        })
    return out