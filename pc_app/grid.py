def generate_grid_points(width_mm, height_mm, spacing_mm):
    if spacing_mm <= 0:
        raise ValueError("grid spacing must be > 0")
    cols = int(width_mm // spacing_mm) + 1
    rows = int(height_mm // spacing_mm) + 1
    points = []
    for r in range(rows):
        y = r * spacing_mm
        if y > height_mm:
            y = height_mm
        for c in range(cols):
            x = c * spacing_mm
            if x > width_mm:
                x = width_mm
            points.append((x, y))
    return points
