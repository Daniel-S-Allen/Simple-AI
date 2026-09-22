from __future__ import annotations

import math
import random
from enum import Enum
from typing import override

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from colorama import Fore, Style

LEARNING_RATE = 0.03
DEAD_NEURON_RATE = 0.1
class Neuron:
    connections: list[Neuron] # connections to the right of this one
    weights: list[float] # weights connecting to the neuron to the right
    biases: list[float] # biases of neurons to the right of this one
    value: float = 0.0
    dead: bool = False
    def __init__(self):
        self.connections = []
        self.weights = []
        self.biases = []
        
    def connect(self, neuron:Neuron, weight:float|None = None, bias:float = 0.1, fan_in:int = 1):
        self.connections.append(neuron)
        if weight is None:
            weight = random.gauss(0, math.sqrt(2.0/fan_in))
        self.weights.append(weight)
        self.biases.append(bias)
        
    def activate(self, value:float = 0.0, method:str = "relu"):
        if self.dead:
            return 0.0
        match(method):
            case "linear":
                return value
            case "relu":
                return value if value > 0 else DEAD_NEURON_RATE*value
            case _:
                raise NotImplementedError("Not implemented. Please use another activation function.")
        
class Network:
    input_layer:list[Neuron]
    hidden_layers:list[list[Neuron]]
    output_layer:list[Neuron]
    def __init__(self, input:int, hidden:list[int], output:int):
        self.input_layer = [Neuron() for _ in range(input)]
        self.hidden_layers = [[Neuron() for _ in range(hidden[i])] for i in range(len(hidden))]
        self.output_layer = [Neuron() for _ in range(output)]
        
        for input_neuron in self.input_layer:
            for hidden_neuron_in_first_layer in self.hidden_layers[0]:
                input_neuron.connect(hidden_neuron_in_first_layer, fan_in=len(self.input_layer))
        
        for hidden_layer_id in range(len(self.hidden_layers)-1):
            for hidden_neuron_in_current_layer in self.hidden_layers[hidden_layer_id]:
                for hidden_neuron_in_next_layer in self.hidden_layers[hidden_layer_id+1]:
                    hidden_neuron_in_current_layer.connect(hidden_neuron_in_next_layer, fan_in=len(self.hidden_layers[hidden_layer_id]))
                    
        for hidden_neuron_in_last_layer in self.hidden_layers[-1]:
            for output_neuron in self.output_layer:
                hidden_neuron_in_last_layer.connect(output_neuron, fan_in=len(self.hidden_layers[-1]))
                
    def forward(self, input_values:list[float]):
        for neuron, value in zip(self.input_layer, input_values):
            neuron.value = value
        def feed_forward(left_layer:list[Neuron], right_layer:list[Neuron], is_output:bool = False):
            sums = [0.0] * len(right_layer)
            for source_neuron in left_layer:
                if not source_neuron.dead:
                    for destination_id, weight in enumerate(source_neuron.weights):
                        sums[destination_id] += source_neuron.value * weight
            
            for source_neuron in left_layer:
                if not source_neuron.dead:
                    for destination_id, bias in enumerate(source_neuron.biases):
                        sums[destination_id] += bias / len(left_layer)
            
            if not is_output:
                for i, neuron in enumerate(right_layer):
                    neuron.value = neuron.activate(sums[i], "relu")
            else:
                # TODO learn how this block of code works
                max_s = max(sums)
                exp_scores = [math.exp(s - max_s) for s in sums]
                sum_exp = sum(exp_scores)
                for target_neuron, exp_s in zip(right_layer, exp_scores):
                    target_neuron.value = exp_s / sum_exp

        current_layer = self.input_layer
        for hidden_layer in self.hidden_layers:
            feed_forward(current_layer, hidden_layer)
            current_layer = hidden_layer
        
        feed_forward(current_layer, self.output_layer, True)
        return [output_neuron.value for output_neuron in self.output_layer]   
                    
    def backpropagate(self, target_label:int, learning_rate:float = 0.75):
        output_delta:list[float] = []
        for id, neuron in enumerate(self.output_layer):
            target = 1.0 if id == target_label else 0.0
            output_delta.append(target - neuron.value)
        
        all_delta: list[list[float]] = []
        current_delta = output_delta
        current_layer = self.output_layer
        
        # Loops backwards over hidden layers
        for layer_id in range(len(self.hidden_layers) - 1, -1, -1):
            hidden_layer = self.hidden_layers[layer_id]
            hidden_delta: list[float] = []
            
            for neuron in hidden_layer:
                error = 0.0
                for target_neuron, weight in zip(neuron.connections, neuron.weights):
                    target_id = current_layer.index(target_neuron)
                    error += current_delta[target_id] * weight
                
                relu_gradient = 1.0 if neuron.value > 0 else DEAD_NEURON_RATE
                hidden_delta.append(error * relu_gradient)
                
            all_delta.insert(0, hidden_delta)
            current_delta = hidden_delta
            current_layer = hidden_layer
        
        all_layers = [self.input_layer] + self.hidden_layers
        all_deltas = all_delta + [output_delta]
        
        for layer_id, left_layer in enumerate(all_layers):
            target_deltas = all_deltas[layer_id]
            for neuron in left_layer:
                for i in range(len(neuron.connections)):
                    delta = target_deltas[i]
                    
                    weight_gradient = neuron.value * delta
                    bias_gradient = delta / len(left_layer)
                    
                    neuron.weights[i] += learning_rate * weight_gradient
                    neuron.biases[i] += learning_rate * bias_gradient
        
    
    def visualize(
        self,
        title: str = "Neural Network Visualization",
        class_names: list[str] | None = None,
        ax: plt.Axes | None = None,
    ):
        """Draws network state onto a specified Matplotlib axis (or creates a new figure if None)."""
        G = nx.DiGraph()
        pos = {}
        node_colors = []
        labels = {}

        layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        layer_colors = ["#4CAF50"] + ["#2196F3"] * len(self.hidden_layers)

        output_values = [n.value for n in self.output_layer]
        predicted_idx = int(np.argmax(output_values)) if output_values else -1

        def is_neuron_dead(neuron: Neuron, layer_idx: int) -> bool:
            """Determines if a neuron is dead (explicitly marked or inactive hidden neuron)."""
            if neuron.dead:
                return True
            # Hidden layer neuron with non-positive activation output
            # if 0 < layer_idx < len(layers) - 1 and neuron.value <= 0:
            #     return True
            return False

        # 1. Assign positions, labels, and colors
        for layer_idx, layer in enumerate(layers):
            n_nodes = len(layer)
            y_offsets = np.linspace(-n_nodes / 2, n_nodes / 2, n_nodes) if n_nodes > 1 else [0]

            for node_idx, neuron in enumerate(layer):
                G.add_node(neuron)
                pos[neuron] = (layer_idx, -y_offsets[node_idx])
                labels[neuron] = f"{neuron.value:.2f}"

                # Color dead neurons black
                if is_neuron_dead(neuron, layer_idx):
                    node_colors.append("#212121")
                elif layer_idx < len(layers) - 1:
                    node_colors.append(layer_colors[layer_idx])
                else:
                    if node_idx == predicted_idx:
                        node_colors.append("#E91E63")  # Predicted Output Highlight
                    else:
                        node_colors.append("#FF9800")

        # 2. Extract edges (Omitting outgoing connections from dead neurons)
        edge_weights = []
        for layer_idx, layer in enumerate(layers[:-1]):
            for neuron in layer:
                # Do not draw outgoing edges if the source neuron is dead
                if is_neuron_dead(neuron, layer_idx):
                    continue

                for target_neuron, weight in zip(neuron.connections, neuron.weights):
                    G.add_edge(neuron, target_neuron, weight=weight)
                    edge_weights.append(weight)

        # Handle Plotting Context
        is_standalone = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            is_standalone = True

        if class_names and 0 <= predicted_idx < len(class_names):
            full_title = f"{title}\nPred: {class_names[predicted_idx]}"
        else:
            full_title = f"{title}\nPred Index: {predicted_idx}"

        ax.set_title(full_title, fontsize=11, fontweight="bold", pad=15)

        # 3. Draw Network Elements
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=850, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="white", font_weight="bold", ax=ax)

        # Only draw edges if active outgoing connections exist
        if edge_weights:
            widths = [1 + 2.5 * abs(w) for w in edge_weights]
            edge_colors = ["#444444" if w >= 0 else "#D32F2F" for w in edge_weights]

            nx.draw_networkx_edges(
                G,
                pos,
                width=widths,
                edge_color=edge_colors,
                arrowsize=10,
                arrowstyle="->",
                connectionstyle="arc3,rad=0.05",
                ax=ax,
            )

        layer_names = ["Input"] + [f"H{i+1}" for i in range(len(self.hidden_layers))] + ["Output"]
        for i, name in enumerate(layer_names):
            ax.text(i, max([p[1] for p in pos.values()]) + 0.7, name, ha="center", fontsize=9, fontweight="semibold")

        if class_names:
            output_x = len(layers) - 1
            for node_idx, neuron in enumerate(self.output_layer):
                if node_idx < len(class_names):
                    y_pos = pos[neuron][1]
                    name_str = class_names[node_idx]
                    is_pred = node_idx == predicted_idx
                    text_color = "#E91E63" if is_pred else "#333333"
                    font_wt = "bold" if is_pred else "normal"
                    ax.text(
                        output_x + 0.25,
                        y_pos,
                        f"← {name_str}",
                        ha="left",
                        va="center",
                        fontsize=8,
                        fontweight=font_wt,
                        color=text_color,
                    )

        ax.axis("off")

        if is_standalone:
            plt.tight_layout()
            plt.show()

    def visualize_multiple(self, samples: list[FruitClass], class_names: list[str] | None = None):
        """Executes forward passes for multiple samples and plots network visualizations side-by-side."""
        n_samples = len(samples)
        fig, axes = plt.subplots(1, n_samples, figsize=(5.5 * n_samples, 6))

        if n_samples == 1:
            axes = [axes]

        for idx, (sample, ax) in enumerate(zip(samples, axes)):
            self.forward([sample.size, sample.weight])
            actual_label = sample.type.name if hasattr(sample, "type") else f"Sample {idx+1}"
            sub_title = f"Test #{idx+1} (Actual: {actual_label})"
            self.visualize(title=sub_title, class_names=class_names, ax=ax)

        plt.tight_layout()
        plt.show()
            
class FRUIT_TYPE(Enum):
    APPLE = 0
    PEAR = 1
    BANANA = 2
    
class FruitClass:
    size: float
    weight: float
    type: FRUIT_TYPE
    def __init__(self, size:float, weight:float, type:FRUIT_TYPE):
        self.size = size
        self.weight = weight
        self.type = type
        
    @override
    def __repr__(self):
        return f"[{self.type.name} (Size: {self.size}, Weight: {self.weight})]"
    
    
def generate_apple() -> FruitClass:
    size = random.uniform(1.0,3.0)
    weight = random.uniform(3.0, 5.0)
    return FruitClass(size, weight, FRUIT_TYPE.APPLE)

def generate_pear() -> FruitClass:
    size = random.uniform(3.5,5.5)
    weight = random.uniform(2.0,3.5)
    return FruitClass(size, weight, FRUIT_TYPE.PEAR)

def generate_banana() -> FruitClass:
    size = random.uniform(6.0,9.0)
    weight = random.uniform(0.5,1.8)
    return FruitClass(size, weight, FRUIT_TYPE.BANANA)

def normalize_dataset(data: list[FruitClass], stats: tuple[float, float, float, float] | None = None):
    """Normalizes dataset features. If stats are provided, uses them to scale without leakage."""
    if stats is None:
        min_s = min(f.size for f in data)
        max_s = max(f.size for f in data)
        min_w = min(f.weight for f in data)
        max_w = max(f.weight for f in data)
        stats = (min_s, max_s, min_w, max_w)
    else:
        min_s, max_s, min_w, max_w = stats

    for f in data:
        f.size = (f.size - min_s) / (max_s - min_s + 1e-8)
        f.weight = (f.weight - min_w) / (max_w - min_w + 1e-8)

    return data, stats

def generate_data(count:int) -> list[FruitClass]:
    data: list[FruitClass] = []
    for _ in range(count):
        f_type = random.choice(list(FRUIT_TYPE))
        match f_type:
            case FRUIT_TYPE.APPLE:
                data.append(generate_apple())
            case FRUIT_TYPE.BANANA:
                data.append(generate_banana())
            case FRUIT_TYPE.PEAR:
                data.append(generate_pear())
    return data
        

if __name__ == "__main__":
    for _ in range(1):
        training_data = generate_data(200)
        training_data, train_stats = normalize_dataset(training_data)
        net = Network(2,[6],3)

        for _ in range(50):
            random.shuffle(training_data)
            for fruit in training_data:
                net.forward([fruit.size, fruit.weight])
                net.backpropagate(fruit.type.value, LEARNING_RATE)
        
        test_data = generate_data(1000)
        test_data, _ = normalize_dataset(test_data, train_stats)
        correct = 0
        for fruit in test_data:
            result = net.forward([fruit.size, fruit.weight])
            predicted = int(np.argmax(result))
            expected = fruit.type.value
            if predicted == expected:
                correct += 1
            # print(f"Probabilities: {[round(p, 3) for p in result]} | Predicted: {predicted}, Expected: {expected}")
        color = Fore.WHITE
        percent_correct = correct/len(test_data)
        if percent_correct < 0.50:
            color = Fore.RED
        elif percent_correct < 0.70:
            color = Fore.LIGHTRED_EX
        elif percent_correct < 0.80:
            color = Fore.YELLOW
        elif percent_correct < 0.95:
            color = Fore.LIGHTBLACK_EX
        
        
        print(f"{color}\nAccuracy: {correct}/{len(test_data)} ({correct / len(test_data) * 100:.1f}%){Style.RESET_ALL}")
        test_samples = [
            next(f for f in test_data if f.type == FRUIT_TYPE.APPLE),
            next(f for f in test_data if f.type == FRUIT_TYPE.PEAR),
            next(f for f in test_data if f.type == FRUIT_TYPE.BANANA),
        ]
        net.visualize_multiple(test_samples, class_names=[f.name for f in FRUIT_TYPE])
        net.hidden_layers[0][3].dead = True
        net.hidden_layers[0][2].dead = True
        net.hidden_layers[0][1].dead = True
        correct = 0
        for fruit in test_data:
            result = net.forward([fruit.size, fruit.weight])
            predicted = int(np.argmax(result))
            expected = fruit.type.value
            if predicted == expected:
                correct += 1
            # print(f"Probabilities: {[round(p, 3) for p in result]} | Predicted: {predicted}, Expected: {expected}")
        color = Fore.WHITE
        percent_correct = correct/len(test_data)
        if percent_correct < 0.50:
            color = Fore.RED
        elif percent_correct < 0.70:
            color = Fore.LIGHTRED_EX
        elif percent_correct < 0.80:
            color = Fore.YELLOW
        elif percent_correct < 0.95:
            color = Fore.LIGHTBLACK_EX
        
        
        print(f"{color}\nAccuracy: {correct}/{len(test_data)} ({correct / len(test_data) * 100:.1f}%){Style.RESET_ALL}")
        net.visualize_multiple(test_samples, class_names=[f.name for f in FRUIT_TYPE])