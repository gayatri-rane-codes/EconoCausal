import os
import sys
import networkx as nx
import pandas as pd
from dowhy import CausalModel


def get_causal_graph_dot() -> str:
    """
    Returns the DOT representation of the validated EconoCausal DAG.
    Only includes verified pre-treatment confounders present in the dataset schema.
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
treatment_arm -> purchased;
}"""


def verify_acyclicity(dot_str: str) -> bool:
    """
    Parses DOT graph into NetworkX DiGraph and strictly verifies acyclicity.
    """
    lines = [line.strip().rstrip(';') for line in dot_str.splitlines() if '->' in line]
    G = nx.DiGraph()
    for line in lines:
        src, dst = line.split('->')
        G.add_edge(src.strip(), dst.strip())

    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"[VERIFIED] Graph is strictly Acyclic (DAG): {is_dag} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return is_dag


def build_and_identify(data_path: str = "data/raw/econocausal_synthetic_dataset.csv"):
    """
    Initializes DoWhy CausalModel with verified DAG and performs Backdoor identification.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    dot_graph = get_causal_graph_dot()

    # Step 1: Verify acyclicity
    if not verify_acyclicity(dot_graph):
        raise ValueError("Provided causal graph contains cycles and is not a valid DAG.")

    # Step 2: Initialize DoWhy causal model
    model = CausalModel(
        data=df,
        treatment="treatment_arm",
        outcome="purchased",
        graph=dot_graph,
    )

    print("\n--- CAUSAL GRAPH VERIFICATION ---")
    print(f"Treatment: {model._treatment}")
    print(f"Outcome: {model._outcome}")

    # Step 3: Identify causal effect using backdoor criterion
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    print("\n--- DOWHY IDENTIFICATION RESULT ---")
    print(identified_estimand)

    backdoor_variables = identified_estimand.get_backdoor_variables()
    print("\n--- IDENTIFIED BACKDOOR ADJUSTMENT SET ---")
    print(f"Backdoor variables to control: {sorted(backdoor_variables)}")

    return model, identified_estimand


if __name__ == "__main__":
    try:
        model, estimand = build_and_identify()
        print("\n[SUCCESS] Causal effect is successfully identifiable under the backdoor criterion.")
    except Exception as e:
        print(f"\n[ERROR] Identification failed: {e}", file=sys.stderr)
        sys.exit(1)
