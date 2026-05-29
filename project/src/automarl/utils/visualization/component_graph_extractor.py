from automarl.component import Component
from automarl.utils.visualization.component_graph import Graph
from automarl.utils.visualization.component_graph import GraphNode
from automarl.utils.visualization.component_graph import GraphEdge
from automarl.components.loggers.logger_component import LoggerSchema


class ComponentGraphExtractor:
    
    """
    Converts a Component tree/forest into a generic Graph model.
    """

    def __init__(
        self,
        include_inputs: bool = True,
        ignore_default_values: bool = True,
        ignore_non_serializable: bool = True,
        max_value_length: int = 120,
        classes_to_ignore=(LoggerSchema),
        input_keys_to_ignore=["artifact_relative_directory", "base_directory", "create_directory", "device", "render_mode"]
    ):

        self.include_inputs = include_inputs

        self.ignore_default_values = ignore_default_values

        self.ignore_non_serializable = ignore_non_serializable

        self.max_value_length = max_value_length

        self._classes_to_ignore = classes_to_ignore
        self._input_keys_to_ignore = input_keys_to_ignore

        self._component_to_id = {}

        self._counter = 0


    def _get_display_name(self, component: Component) -> str:

        if component.name is not None:
            return component.name

        return type(component).__name__

    def _extract_attributes(
        self,
        component: Component,
        graph : Graph
    ) -> dict[str, str]:

        if not self.include_inputs:
            return {}

        attributes = {}

        input_meta = component.get_input_meta()

        for key, value in component.input.items():

            if key in self._input_keys_to_ignore:
                continue

            parameter_signature = component.get_parameter_signature(key)

            if parameter_signature is None:
                continue

            if self.ignore_non_serializable and input_meta[key].ignore_at_serialization:
                continue

            if self.ignore_default_values and not input_meta[key].was_custom_value_passed():
                continue

            if isinstance(value, Component):

                if isinstance(value, self._classes_to_ignore):
                    continue

                if value in component.child_components:
                    continue

                graph.add_edge(
                    GraphEdge(
                        source=self._get_id(component),
                        target=self._get_id(value),
                        edge_type=GraphEdge.REFERENCE_TYPE                       
                    )
                )

            if isinstance(value, list):

                if len(value) > 0 and isinstance(value[0], Component):
                    continue

            attributes[key] = self._serialize_value(value)

        return attributes
    
    def _serialize_value(self, value):
    
        if value is None:
            return "None"
    
        if isinstance(value, (int, float, bool)):
            return str(value)
    
        if isinstance(value, str):
        
            if len(value) > self.max_value_length:
                return value[:self.max_value_length] + "..."
    
            return value
    
        if isinstance(value, list):
            return f"list[{len(value)}]"
    
        if isinstance(value, tuple):
            return f"tuple[{len(value)}]"
    
        if isinstance(value, dict):
            return f"dict[{len(value)}]"
    
        return type(value).__name__


    def extract(self, roots: list[Component]) -> Graph:

        graph = Graph()

        for root in roots:
            self._visit_for_child_parent_relations(root, graph, set())
            self._visit_for_attributes(root, graph, set())

        return graph
    


    def _get_id(self, component) -> str:
        cid = id(component)

        if cid not in self._component_to_id:
            self._component_to_id[cid] = f"n{self._counter}"
            self._counter += 1

        return self._component_to_id[cid]
    

    def _visit_for_child_parent_relations(self, component, graph: Graph, visited: set[int]):

        cid = id(component)

        node_id = self._get_id(component)

        if cid in visited:
            return

        visited.add(cid)

        for child in getattr(component, "child_components", []):

            if isinstance(child, self._classes_to_ignore):
                continue
                    
            child_id = self._get_id(child)

            graph.add_edge(
                GraphEdge(
                    source=node_id,
                    target=child_id,
                    edge_type=GraphEdge.PARENT_CHILD_TYPE,
                )
            )

            self._visit_for_child_parent_relations(child, graph, visited)

    def _visit_for_attributes(self, component, graph: Graph, visited: set[int]):

        cid = id(component)
    
        node_id = self._get_id(component)

        if cid not in visited:
            
            graph.add_node(
                GraphNode(
                    id=node_id,
                    display_name=self._get_display_name(component),
                    class_name=type(component).__name__,
                    attributes=self._extract_attributes(component, graph),
                    data=component,
                )
            )

        visited.add(cid)

        for child in getattr(component, "child_components", []):

            if isinstance(child, self._classes_to_ignore):
                continue

            self._visit_for_attributes(child, graph, visited)
