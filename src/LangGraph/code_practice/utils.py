import os


def generate_mermaid_images(graph):
    current_filename = os.path.splitext(os.path.basename(__file__))[0]
    draw_mermaid_path = os.path.join(
        os.path.dirname(__file__), "../mermaid_images", f"{current_filename}.png"
    )
    graph.get_graph().draw_mermaid_png(output_file_path=draw_mermaid_path)
