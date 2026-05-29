


from automarl.utils.visualization.component_graph_renderer import GraphRenderer

CODE_PATH = "mermaid_flow_chart_mermaid.md"


class MermaidRenderer(GraphRenderer):

    def _save_render_to_file(self, rendered_graph, path_to_save):

        with open(path_to_save, "w", encoding="utf-8") as f:
            f.write(rendered_graph)


    @staticmethod
    def _escape(text: str) -> str:

        if text is None:
            return ""

        text = str(text)

        replacements = {
            '"': "#quot;",
            "'": "\\'",
            "\n": "<br/>",
            "\t": "    ",
            "{": "(",
            "}": ")",
            "[": "(",
            "]": ")",
            "<": "&lt;",
            ">": "&gt;",
        }

        for original, replacement in replacements.items():
            text = text.replace(original, replacement)

        return text