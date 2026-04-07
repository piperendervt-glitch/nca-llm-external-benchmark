# NCA-LLM External Benchmark Verification

## Overview
Verification phase of NCA-LLM research.
Uses external human-designed benchmarks
to validate findings from exploration phase.

## Exploration Phase (archived)
Repository: nca-llm-shared-premises
Finding: Diversity effect observed in
Claude-generated world_consistency tasks (+16.3pp CFR diff)
Limitation: Task creator (Claude) and solver (LLMs)
in same ecosystem → external validity unconfirmed

## Verification Phase (this repo)
Benchmarks: COPA, αNLI, RuleTaker-depth-1
Conditions:
  - 7b_homo_NCA
  - 7b_het_NCA (qwen2.5/llama3/mistral)
  - 7b_het_NCA (qwen2.5/gemma2:9b/phi3)
  - 7b_majority_vote
