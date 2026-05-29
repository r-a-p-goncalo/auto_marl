from automarl.utils.visualization.mermaid.mermaid_renderer import MermaidRenderer
from automarl.utils.visualization.component_graph import Graph


class MermaidFlowchartRenderer(MermaidRenderer):

    PATH_TO_SAVE = "flowchart_mermaid.md"

    def __init__(
            
        self,
        show_component_type: bool = True,
        show_component_name: bool = True,
        show_children_count: bool = False,
        include_input_keys: bool = False
            
        ):
        self.header = "flowchart TD"

    def render(self, graph: Graph) -> str:
        lines = [self.header]

        for node in graph.nodes.values():
            label = self._escape(node.label)
            lines.append(f'{node.id}["{label}"]')

        for edge in graph.edges:
            lines.append(f"{edge.source} --> {edge.target}")

        return "\n".join(lines)

    def _escape(self, text: str) -> str:
        return text.replace('"', "#quot;").replace("\n", "<br/>")
    