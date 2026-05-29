from dataclasses import dataclass, field
from typing import Any

from automarl.component import Component


@dataclass
class GraphNode:
    
    id: str
    label: str
    type_name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    data: Any = None


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str | None = None


@dataclass
class Graph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)



    

