# IA_integration/prompt_builder.py

from __future__ import annotations

from typing import Dict, Optional


SYSTEM_PROMPT = """
You are an AI assistant embedded inside a professional Synthetic Control application.

Your role:
- explain the current configuration
- help interpret charts and estimation results
- explain robustness diagnostics
- remain strictly consistent with the provided application context
- never invent numerical outputs not present in the context
- never pretend to have recomputed SCM
- when uncertainty exists, say so clearly
""".strip()


def build_prompt(
    task: str,
    context: Dict,
    user_message: Optional[str] = None,
) -> str:
    """
    Build a structured prompt for the SCM assistant.
    """
    context_block = repr(context)

    instructions = {
        "data_config": (
            "Explain the current data and SCM setup. "
            "Highlight possible risks in treatment unit choice, donor pool, "
            "time window, and covariates."
        ),
        "exploration": (
            "Interpret the exploration context. "
            "Point out missing data risks, comparability issues, and any warning signals."
        ),
        "estimation": (
            "Explain the selected estimation method and its implications. "
            "Clarify the role of optimization settings and expected trade-offs."
        ),
        "results": (
            "Interpret the estimation results comprehensively. "
            "Assess pre-treatment fit quality: how well does the synthetic control track the treated unit? "
            "Comment on post-treatment gap: is there a clear, sustained causal effect? "
            "Evaluate donor weight concentration (are 1-2 donors dominating?). "
            "Mention key methodological cautions: parallel trends assumption, "
            "interpolation bias if weights are extreme, and limits of observational causal inference."
        ),
        "fit_quality": (
            "Analyse the pre-treatment fit quality of the synthetic control. "
            "A good pre-treatment fit (low pre-RMSPE, visual overlap) is necessary for valid causal inference. "
            "Comment on any visible divergences before T0. "
            "If fit quality is poor, explain what this means for the validity of post-T0 estimates."
        ),
        "donor_weights": (
            "Interpret the donor unit weights. "
            "Identify which donors contribute most and whether this is plausible. "
            "Flag concentration risk: if one donor has weight > 0.5, the synthetic control mimics a single unit. "
            "Comment on whether high-weight donors are economically comparable to the treated unit."
        ),
        "rmspe_ratio": (
            "Evaluate the significance of the RMSPE ratio (post-RMSPE / pre-RMSPE). "
            "A ratio >> 1 suggests the post-treatment divergence is large relative to pre-treatment noise, "
            "consistent with a real causal effect. "
            "In placebo tests, if the treated unit's ratio ranks at the top of the distribution, "
            "this provides informal statistical evidence. "
            "Interpret the ratio in context, and state clearly whether the result "
            "supports or challenges the treatment effect hypothesis."
        ),
        "robustness": (
            "Interpret robustness outputs systematically. "
            "1. Placebo test: where does the treated unit rank by RMSPE ratio? "
            "A p-value < 0.10 (rank / total) is suggestive; < 0.05 is conventional significance. "
            "2. Leave-one-out (LOO): do results hold when major donors are excluded? "
            "Stability across LOO runs strengthens causal claims. "
            "3. Backdating: does the placebo treatment applied before T0 produce false positives? "
            "False alarms reduce credibility. "
            "4. Synthesis: integrate the three tests into an overall credibility judgment."
        ),
        "free": (
            "Answer the user's question using only the application context."
        ),
        "exploration_chart_comment": (
            "Interpret a single exploration chart from the provided structured context. "
            "Describe what is directly visible, highlight data quality or comparability issues, "
            "and provide a cautious analytical comment suitable for a professional SCM workflow."
        ),
    }

    task_instruction = instructions.get(task, instructions["free"])

    user_block = user_message.strip() if user_message else "No extra user question."

    return f"""
{SYSTEM_PROMPT}

Task:
{task_instruction}

Application context:
{context_block}

User message:
{user_block}

Response requirements:
- be precise
- be structured
- be concise but informative
- do not fabricate values
- clearly distinguish observation from interpretation
""".strip()