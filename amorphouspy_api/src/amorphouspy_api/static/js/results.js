// Global variables for 3D visualization
let viewer;
let isolatedAtom = null; // tracks whether we're in isolation mode

// Initialize when page loads
window.onload = function () {
    initPlotlyPlot();
    init3DViewer();
};

// Render the Plotly plot
function initPlotlyPlot() {
    // Use the height from the figure layout (set in Python) for the container
    const figHeight = plotlyData.layout && plotlyData.layout.height ? plotlyData.layout.height : 1500;
    const plotDiv = document.getElementById('plotly-div');
    plotDiv.style.height = figHeight + 'px';

    Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
        displaylogo: false
    });
}

// Initialize 3D structure viewer
function init3DViewer() {
    console.log('Structure data length:', structureData.length);
    console.log('Structure data preview:', structureData.substring(0, 200));

    const containerDiv = document.getElementById('structure-div');
    if (!containerDiv) {
        console.error('Structure container div not found');
        return;
    }

    if (!structureData.trim()) {
        containerDiv.innerHTML =
            '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;">No structure data available</div>';
        return;
    }

    // Clear any existing content
    containerDiv.innerHTML = '';

    viewer = $3Dmol.createViewer(containerDiv, {
        hoverDuration: 250  // Try to reduce hover delay (if supported)
    });

    // Add model using extended XYZ format which includes cell information
    viewer.addModel(structureData, 'xyz');

    // Set initial style with hover labels
    viewer.setStyle({}, {
        sphere: { radius: 0.5, colorscheme: 'Jmol' },
        stick: { radius: 0.2, colorscheme: 'Jmol' }
    });

    // Enable hover detection for all atoms
    viewer.setHoverable({}, true, function (atom, viewer, event, container) {
        if (!atom.label) {
            // Get element symbol and atom index
            const element = atom.elem || atom.element || 'Unknown';
            const atomIndex = atom.serial || (atom.index !== undefined ? atom.index + 1 : 'N/A');

            atom.label = viewer.addLabel(
                `${element} (${atomIndex})`,
                {
                    position: atom,
                    backgroundColor: 'black',
                    backgroundOpacity: 0.8,
                    fontColor: 'white',
                    fontSize: 12,
                    borderThickness: 1,
                    borderColor: 'black'
                }
            );
        }
    }, function (atom) {
        if (atom.label) {
            viewer.removeLabel(atom.label);
            delete atom.label;
        }
    });

    // Enable click-to-isolate: clicking an atom hides everything beyond 5 Å
    viewer.setClickable({}, true, function (atom) {
        isolateAtom(atom);
    });

    // Add unit cell visualization if available
    addUnitCell();

    viewer.setBackgroundColor('white');
    viewer.zoomTo();
    viewer.render();
    buildLegend();
    console.log('3D viewer initialized successfully');
}

// Build a color legend from unique elements in the loaded structure
function buildLegend() {
    const legendDiv = document.getElementById('structure-legend');
    if (!legendDiv || !viewer) return;

    // Extract unique elements from the XYZ data
    const lines = structureData.trim().split('\n');
    const numAtoms = parseInt(lines[0], 10);
    const elements = new Set();
    for (let i = 2; i < 2 + numAtoms && i < lines.length; i++) {
        const parts = lines[i].trim().split(/\s+/);
        if (parts.length > 0) elements.add(parts[0]);
    }

    if (elements.size === 0) return;

    // Use Jmol colors — same scheme set in viewer styles
    const colorMap = ($3Dmol.elementColors && $3Dmol.elementColors.Jmol) || {};
    legendDiv.innerHTML = '';
    const sorted = Array.from(elements).sort();
    sorted.forEach(function (elem) {
        const colorVal = colorMap[elem];
        let css = '#aaaaaa';
        if (colorVal !== undefined && colorVal !== null) {
            if (typeof colorVal === 'number') {
                css = '#' + ('000000' + colorVal.toString(16)).slice(-6);
            } else if (typeof colorVal === 'string') {
                css = colorVal;
            }
        }
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML =
            '<span class="legend-swatch" style="background:' + css + '"></span>' + elem;
        legendDiv.appendChild(item);
    });
}

// Add unit cell visualization for extended XYZ format
function addUnitCell() {
    try {
        // Parse the extended XYZ format to extract lattice parameters
        const lines = structureData.trim().split('\n');

        if (lines.length < 2) return;

        // Extended XYZ format has lattice info in the comment line (line 1)
        const commentLine = lines[1];

        // Look for Lattice parameter in the comment line
        const latticeMatch = commentLine.match(/Lattice="([^"]+)"/);

        if (!latticeMatch) {
            console.log('No lattice information found in extended XYZ');
            return;
        }

        // Parse the lattice vectors from the string
        const latticeValues = latticeMatch[1].split(/\s+/).map(parseFloat);

        if (latticeValues.length !== 9) {
            console.log('Invalid lattice format');
            return;
        }

        // Create lattice vectors as 3x3 matrix
        const a = [latticeValues[0], latticeValues[1], latticeValues[2]];
        const b = [latticeValues[3], latticeValues[4], latticeValues[5]];
        const c = [latticeValues[6], latticeValues[7], latticeValues[8]];

        // Define the 8 corners of the unit cell
        const origin = [0, 0, 0];
        const corners = [
            [0, 0, 0],
            a,
            [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            b,
            c,
            [a[0] + c[0], a[1] + c[1], a[2] + c[2]],
            [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]],
            [b[0] + c[0], b[1] + c[1], b[2] + c[2]]
        ];

        // Define the 12 edges of the unit cell
        const edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], // bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], // top face
            [0, 4], [1, 5], [2, 6], [3, 7]  // vertical edges
        ];

        // Add lines for each edge
        edges.forEach(([i, j]) => {
            viewer.addLine({
                start: { x: corners[i][0], y: corners[i][1], z: corners[i][2] },
                end: { x: corners[j][0], y: corners[j][1], z: corners[j][2] },
                color: 'black',
                radius: 0.1
            });
        });

        console.log('Unit cell added successfully');

    } catch (error) {
        console.log('Error adding unit cell:', error);
    }
}

// Update visualization style
function updateStyle() {
    if (!viewer) return;

    const style = document.getElementById('style-select').value;

    // Clear only the molecular model, keep unit cell
    viewer.removeAllModels();
    viewer.addModel(structureData, 'xyz');

    // Apply the selected style
    viewer.setStyle({}, getStyleForMode(style));

    // If in isolation mode, re-apply isolation
    if (isolatedAtom) {
        isolateAtom(isolatedAtom);
    }

    // Re-add hover functionality
    viewer.setHoverable({}, true, function (atom, viewer, event, container) {
        if (!atom.label) {
            // Get element symbol and atom index
            const element = atom.elem || atom.element || 'Unknown';
            const atomIndex = atom.serial || (atom.index !== undefined ? atom.index + 1 : 'N/A');

            atom.label = viewer.addLabel(
                `${element} (${atomIndex})`,
                {
                    position: atom,
                    backgroundColor: 'black',
                    backgroundOpacity: 0.8,
                    fontColor: 'white',
                    fontSize: 12,
                    borderThickness: 1,
                    borderColor: 'black'
                }
            );
        }
    }, function (atom) {
        if (atom.label) {
            viewer.removeLabel(atom.label);
            delete atom.label;
        }
    });

    // Re-add click-to-isolate
    viewer.setClickable({}, true, function (atom) {
        isolateAtom(atom);
    });

    // Re-add unit cell (unit cell shapes persist across model changes)
    addUnitCell();

    viewer.render();
}

// Reset camera view
function resetView() {
    if (viewer) {
        viewer.zoomTo();
        viewer.render();
    }
}

// Isolate atoms within 5 Å of the clicked atom
function isolateAtom(atom) {
    if (!viewer) return;
    const cx = atom.x, cy = atom.y, cz = atom.z;
    const radius = 5.0;
    const allAtoms = viewer.getModel().selectedAtoms({});
    const nearIndices = [];

    allAtoms.forEach(function (a) {
        const dx = a.x - cx, dy = a.y - cy, dz = a.z - cz;
        if (dx * dx + dy * dy + dz * dz <= radius * radius) {
            nearIndices.push(a.index);
        }
    });

    // Hide all atoms, then show only nearby ones with the current style
    viewer.setStyle({}, { sphere: { hidden: true }, stick: { hidden: true } });

    const style = document.getElementById('style-select').value;
    const visStyle = getStyleForMode(style);
    nearIndices.forEach(function (idx) {
        viewer.setStyle({ index: idx }, visStyle);
    });

    isolatedAtom = atom;
    document.getElementById('show-all-btn').style.display = 'inline-block';

    viewer.render();
}

// Show all atoms again
function showAllAtoms() {
    if (!viewer) return;
    const style = document.getElementById('style-select').value;
    viewer.setStyle({}, getStyleForMode(style));
    isolatedAtom = null;
    document.getElementById('show-all-btn').style.display = 'none';
    viewer.render();
}

// Return the 3Dmol style object for a given mode name
function getStyleForMode(mode) {
    switch (mode) {
        case 'sphere':
            return { sphere: { radius: 0.5, colorscheme: 'Jmol' }, stick: { radius: 0.2, colorscheme: 'Jmol' } };
        case 'stick':
            return { stick: { radius: 0.3, colorscheme: 'Jmol' } };
        case 'line':
            return { line: { linewidth: 2, colorscheme: 'Jmol' } };
        case 'cross':
            return { cross: { radius: 0.5, colorscheme: 'Jmol' } };
        default:
            return { sphere: { radius: 0.5, colorscheme: 'Jmol' }, stick: { radius: 0.2, colorscheme: 'Jmol' } };
    }
}

// Download functionality for plots
function downloadPlot(format) {
    const filename = `glass_analysis_${taskId}`;
    if (format === 'png') {
        Plotly.downloadImage('plotly-div', {
            format: 'png',
            width: 1400,
            height: 1200,
            filename: filename
        });
    } else if (format === 'svg') {
        Plotly.downloadImage('plotly-div', {
            format: 'svg',
            width: 1400,
            height: 1200,
            filename: filename
        });
    }
}
