def update_layout(layout, obstacles):
    for cell in obstacles:
        layout.mark_obstructed(cell)