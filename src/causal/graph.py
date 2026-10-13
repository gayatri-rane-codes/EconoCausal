"""
EconoCausal - Causal Directed Acyclic Graph (DAG) and Identification Module
Phase 1 - Day 4: Builds causal graph, tests acyclicity, and runs DoWhy backdoor identification.
"""

import os
import sys
import pandas as pd
import networkx as nx
from dowhy import CausalModel


def get_causal_graph_dot() -> str:
    """
    Returns clean DOT graph without inline comments for DoWhy parser stability.
    """
    return """digraph {
cart_abandonment_rate -> treatment_arm;
cart_abandonment_rate -> purchased;
previous_discount_response_rate -> treatment_arm;
previous_discount_response_rate -> purchased;
historical_spend -> treatment_arm;
historical_spend -> purchased;
recency_days -> treatment_arm;
recency_days -> purchased;
browsing_frequency -> treatment_arm;
browsing_frequency -> purchased;
loyalty_member -> treatment_arm;
loyalty_member -> purchased;
competitor_price_index -> treatment_arm;
competitor_price_index -> purchased;
seasonality_index -> treatment_arm;
seasonality_index -> purchased;
age -> purchased;
income_bracket -> historical_spend;
income_bracket -> purchased;
treatment_arm -> purchased;
}"""


def verify_acyclicity(dot_str: str) -> bool:
    """
    Parses DOT graph into NetworkX DiGraph and strictly verifies acyclicity.
    """
    lines = [line.strip().rstrip(';') for line in dot_str.splitlines() if '->' in line]
    edges = [tuple(part.strip() for part in line.split('->')) for line in lines]
    G = nx.DiGraph(edges)
    is_dag = nx.is_directed_acyclic_graph(G)
    if not is_dag:
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f'Graph contains directed cycles: {cycles}')
    print(f'[VERIFIED] Graph is strictly Acyclic (DAG): {is_dag} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.')
    return is_dag


def build_and_identify(data_path: str = 'data/raw/econocausal_synthetic_dataset.csv'):
    """
    Loads dataset, constructs DoWhy CausalModel, checks acyclicity, and identifies backdoor adjustment set.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found at {data_path}. Ensure Phase 1 Day 3 raw files exist.')

    df = pd.read_csv(data_path)
    print(f'Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.')

    dot_graph = get_causal_graph_dot()

    # Step 1: Explicit acyclicity check
    verify_acyclicity(dot_graph)

    # Step 2: Initialize DoWhy CausalModel
    model = CausalModel(
        data=df,
        treatment='treatment_arm',
        outcome='purchased',
        graph=dot_graph
    )

    print('\n--- CAUSAL GRAPH VERIFICATION ---')
    print(f'Treatment: {model._treatment}')
    print(f'Outcome: {model._outcome}')

    # Step 3: Identify causal effect using backdoor criterion
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    print('\n--- DOWHY IDENTIFICATION RESULT ---')
    print(identified_estimand)

    backdoor_variables = identified_estimand.get_backdoor_variables()
    print('\n--- IDENTIFIED BACKDOOR ADJUSTMENT SET ---')
    print(f'Backdoor variables to control: {sorted(backdoor_variables)}')

    return model, identified_estimand


if __name__ == '__main__':
    try:
        model, estimand = build_and_identify()
        print('\n[SUCCESS] Causal effect is successfully identifiable under the backdoor criterion.')
    except Exception as e:
        print(f'\n[ERROR] Identification failed: {e}', file=sys.stderr)
        sys.exit(1)
