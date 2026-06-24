import os

"""
图的可视化：
graph.get_graph()

1. draw_mermaid_png 直接生成图片
2. draw_mermaid 生成code 可以在mermaid在线预览器中打开查看
"""


def generate_mermaid_images(graph):
    current_filename = os.path.splitext(os.path.basename(__file__))[0]
    draw_mermaid_path = os.path.join(
        os.path.dirname(__file__), "../mermaid_images", f"{current_filename}.png"
    )
    graph.get_graph().draw_mermaid_png(output_file_path=draw_mermaid_path)
