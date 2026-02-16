import random
import math

def generate_logo_svg(filepath):
    width = 200
    height = 200
    cx, cy = width / 2, height / 2
    r_outer = 80
    
    # Amorphous network parameters
    num_nodes = 25
    nodes = []
    
    # Generate random nodes within a circle
    for _ in range(num_nodes):
        r = math.sqrt(random.uniform(0, 1)) * r_outer
        theta = random.uniform(0, 2 * math.pi)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        nodes.append((x, y))
        
    # Generate edges based on distance
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < 50: # Connectivity threshold
                edges.append((nodes[i], nodes[j]))

    # SVG Colors (Indigo/Blue theme)
    color_edges = "#3949ab" # Indigo lighten-1
    color_nodes = "#1a237e" # Indigo darken-4
    color_accent = "#ff6d00" # Amber accent
    
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
    <!-- Background circle (optional/subtle) -->
    <circle cx="{cx}" cy="{cy}" r="{r_outer + 10}" fill="#e8eaf6" opacity="0.5" />
    
    <!-- Edges -->
    <g stroke="{color_edges}" stroke-width="2" opacity="0.7">'''
    
    for (node1, node2) in edges:
        svg_content += f'<line x1="{node1[0]:.1f}" y1="{node1[1]:.1f}" x2="{node2[0]:.1f}" y2="{node2[1]:.1f}" />'
        
    svg_content += '''
    </g>
    
    <!-- Nodes -->
    <g fill="{color_nodes}">'''
    
    for i, (x, y) in enumerate(nodes):
        # Randomly color some nodes accent
        fill = color_accent if random.random() < 0.2 else color_nodes
        radius = random.uniform(3, 5)
        svg_content += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" />'
        
    svg_content += '''
    </g>
</svg>'''

    with open(filepath, 'w') as f:
        f.write(svg_content)
    print(f"Generated logo at {filepath}")

if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/logo.svg"
    generate_logo_svg(output_path)
