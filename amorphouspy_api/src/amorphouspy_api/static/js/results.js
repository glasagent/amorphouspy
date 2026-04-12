// Global variables for 3D visualization
let viewer;
let isolatedAtom = null; // tracks whether we're in isolation mode
let atomMeta = [];       // per-atom metadata parsed from extxyz: {role, o_class}
let formerCutoffs = {};  // former element -> cutoff in Å from the analysis

// ───── Glass-science color scheme ─────
// Jmol base for formers; custom palette for oxygen classes; modifiers keep Jmol.
const O_CLASS_COLORS = {
    BO: 0xff0d0d,  // standard red - bridging oxygen
    NBO: 0xff69b4,  // pink - non-bridging oxygen
    free: 0x20cc20,  // green - free oxygen (rare)
    tri: 0xff8c00,  // orange - triclustered oxygen
};
const O_DEFAULT_COLOR = 0xff0d0d; // fallback for unlabelled O

// Role-based sphere radii (Å)
const RADII = { former: 0.35, oxygen: 0.40, modifier: 0.65, other: 0.45 };
const STICK_RADIUS = 0.15;

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

    // Parse per-atom metadata from the extended XYZ before handing to 3Dmol
    parseExtXYZMeta();

    // Clear any existing content
    containerDiv.innerHTML = '';

    viewer = $3Dmol.createViewer(containerDiv, {
        hoverDuration: 250
    });

    // Add model — 3Dmol will parse the basic xyz (element + xyz coords)
    viewer.addModel(structureData, 'xyz');

    // Apply default glass-science styles (all atoms)
    applyGlassStyles(false);

    // Enable hover detection for all atoms
    viewer.setHoverable({}, true, function (atom, viewer, event, container) {
        if (!atom.label) {
            const element = atom.elem || atom.element || 'Unknown';
            const atomIndex = atom.serial || (atom.index !== undefined ? atom.index + 1 : 'N/A');
            const meta = atomMeta[atom.index] || {};
            let tip = element + ' (' + atomIndex + ')';
            if (meta.o_class) tip += ' [' + meta.o_class + ']';
            else if (meta.role) tip += ' [' + meta.role + ']';

            atom.label = viewer.addLabel(tip, {
                position: atom,
                backgroundColor: 'black',
                backgroundOpacity: 0.8,
                fontColor: 'white',
                fontSize: 12,
                borderThickness: 1,
                borderColor: 'black'
            });
        }
    }, function (atom) {
        if (atom.label) {
            viewer.removeLabel(atom.label);
            delete atom.label;
        }
    });

    // Enable click-to-isolate
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

// ───── Extended XYZ metadata parser ─────
// Parses the Properties header and extra per-atom columns to build atomMeta[]
// and formerCutoffs from the comment line.
function parseExtXYZMeta() {
    atomMeta = [];
    formerCutoffs = {};
    const lines = structureData.trim().split('\n');
    if (lines.length < 3) return;

    const numAtoms = parseInt(lines[0], 10);
    const commentLine = lines[1];

    // Parse former_cutoffs from comment line
    const cutoffMatch = commentLine.match(/former_cutoffs="(\{[^"]*\})"/);
    if (cutoffMatch) {
        try { formerCutoffs = JSON.parse(cutoffMatch[1].replace(/'/g, '"')); }
        catch (e) { console.log('Could not parse former_cutoffs:', e); }
    }

    // Parse Properties= header to find column layout
    // e.g. Properties=species:S:1:pos:R:3:role:S:1:o_class:S:1
    const propsMatch = commentLine.match(/Properties=([^\s]+)/);
    if (!propsMatch) return;

    const propDefs = propsMatch[1].split(':');
    // Parse triplets: name, type, count
    const columns = [];
    for (let i = 0; i < propDefs.length; i += 3) {
        const name = propDefs[i];
        const count = parseInt(propDefs[i + 2], 10) || 1;
        columns.push({ name: name, count: count });
    }

    // Find the column indices for role and o_class
    let roleCol = -1, oClassCol = -1;
    let colIdx = 0;
    for (const col of columns) {
        if (col.name === 'role') roleCol = colIdx;
        if (col.name === 'o_class') oClassCol = colIdx;
        colIdx += col.count;
    }

    // Parse per-atom lines
    for (let i = 2; i < 2 + numAtoms && i < lines.length; i++) {
        const parts = lines[i].trim().split(/\s+/);
        const meta = { role: '', o_class: '' };
        if (roleCol >= 0 && roleCol < parts.length) meta.role = parts[roleCol];
        if (oClassCol >= 0 && oClassCol < parts.length) meta.o_class = parts[oClassCol];
        atomMeta.push(meta);
    }
}

// ───── Apply glass-science styles ─────
// Custom per-group colors and radii based on role/o_class.
// Former-O bonds are drawn via 3Dmol's built-in stick renderer (GPU-accelerated).
// If networkOnly=true, modifiers and "other" atoms are hidden.
function applyGlassStyles(networkOnly) {
    if (!viewer) return;
    var allAtoms = viewer.getModel().selectedAtoms({});
    var hasMetadata = atomMeta.length === allAtoms.length && atomMeta.some(function (m) { return m.role; });
    var Jmol = ($3Dmol.elementColors && $3Dmol.elementColors.Jmol) || {};

    if (!hasMetadata) {
        // Fallback: no metadata available, use plain Jmol Ball & Stick
        viewer.setStyle({}, {
            sphere: { radius: 0.5, colorscheme: 'Jmol' },
            stick: { radius: 0.2, colorscheme: 'Jmol' }
        });
        return;
    }

    // Assign only former-O bonds into the model (replaces auto-detected bonds)
    assignGlassBonds(allAtoms);

    // Group atom indices by (role, o_class, element) for batched setStyle calls
    var groups = {};  // key -> {indices:[], color:number, radius:number, hasStick:boolean, hidden:boolean}

    allAtoms.forEach(function (a, idx) {
        var meta = atomMeta[idx] || {};
        var elem = a.elem || a.element || '';
        var color, radius, key, hasStick, hidden;

        if (meta.role === 'oxygen') {
            var cls = meta.o_class || 'default';
            key = 'o_' + cls;
            color = O_CLASS_COLORS[cls] || O_DEFAULT_COLOR;
            radius = RADII.oxygen;
            hasStick = true;
            hidden = false;
        } else if (meta.role === 'former') {
            key = 'f_' + elem;
            color = Jmol[elem] || 0x909090;
            radius = RADII.former;
            hasStick = true;
            hidden = false;
        } else if (meta.role === 'modifier') {
            key = 'm_' + elem;
            color = Jmol[elem] || 0x909090;
            radius = RADII.modifier;
            hasStick = false;
            hidden = networkOnly; // hide in network-only mode
        } else {
            key = 'x_' + elem;
            color = Jmol[elem] || 0x909090;
            radius = RADII.other;
            hasStick = false;
            hidden = networkOnly; // hide in network-only mode
        }

        if (!groups[key]) groups[key] = { indices: [], color: color, radius: radius, hasStick: hasStick, hidden: hidden };
        groups[key].indices.push(idx);
    });

    // One setStyle call per group — sticks on formers+oxygens, spheres on all
    Object.keys(groups).forEach(function (key) {
        var g = groups[key];
        if (g.hidden) {
            viewer.setStyle({ index: g.indices }, { sphere: { hidden: true }, stick: { hidden: true } });
        } else {
            var style = { sphere: { radius: g.radius, color: g.color } };
            if (g.hasStick) {
                style.stick = { radius: STICK_RADIUS, color: 0x888888 };
            }
            viewer.setStyle({ index: g.indices }, style);
        }
    });
}

// ───── Fast bond assignment using spatial grid ─────
// Replaces all auto-detected bonds with only former-O bonds.
// Uses a cell list for O(N) performance instead of O(N²).
function assignGlassBonds(allAtoms) {
    var maxCutoff = Math.max(2.0, ...Object.values(formerCutoffs).map(Number));
    var cellSize = maxCutoff;

    // Clear all auto-detected bonds
    allAtoms.forEach(function (a) { a.bonds = []; a.bondOrder = []; });

    // Build spatial grid of oxygen positions
    var oxygensByCell = {};
    allAtoms.forEach(function (a, idx) {
        if ((atomMeta[idx] || {}).role !== 'oxygen') return;
        var key = Math.floor(a.x / cellSize) + ',' +
            Math.floor(a.y / cellSize) + ',' +
            Math.floor(a.z / cellSize);
        if (!oxygensByCell[key]) oxygensByCell[key] = [];
        oxygensByCell[key].push(idx);
    });

    // For each former, search only neighboring cells for oxygens
    allAtoms.forEach(function (a, fi) {
        if ((atomMeta[fi] || {}).role !== 'former') return;
        var fElem = a.elem || a.element || '';
        var cutoff = formerCutoffs[fElem] || maxCutoff;
        var cutSq = cutoff * cutoff;
        var cx = Math.floor(a.x / cellSize);
        var cy = Math.floor(a.y / cellSize);
        var cz = Math.floor(a.z / cellSize);

        for (var dx = -1; dx <= 1; dx++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dz = -1; dz <= 1; dz++) {
                    var oxygens = oxygensByCell[(cx + dx) + ',' + (cy + dy) + ',' + (cz + dz)];
                    if (!oxygens) continue;
                    for (var k = 0; k < oxygens.length; k++) {
                        var oi = oxygens[k];
                        var oa = allAtoms[oi];
                        var ddx = a.x - oa.x, ddy = a.y - oa.y, ddz = a.z - oa.z;
                        if (ddx * ddx + ddy * ddy + ddz * ddz <= cutSq) {
                            a.bonds.push(oi);
                            a.bondOrder.push(1);
                            oa.bonds.push(fi);
                            oa.bondOrder.push(1);
                        }
                    }
                }
            }
        }
    });
}

// Build a color legend from the glass-science coloring
function buildLegend() {
    const legendDiv = document.getElementById('structure-legend');
    if (!legendDiv || !viewer) return;

    const allAtoms = viewer.getModel().selectedAtoms({});
    const hasMetadata = atomMeta.length === allAtoms.length && atomMeta.some(function (m) { return m.role; });
    const Jmol = ($3Dmol.elementColors && $3Dmol.elementColors.Jmol) || {};

    legendDiv.innerHTML = '';

    function addSwatch(color, label) {
        var css;
        if (typeof color === 'number') {
            css = '#' + ('000000' + color.toString(16)).slice(-6);
        } else {
            css = color;
        }
        var item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = '<span class="legend-swatch" style="background:' + css + '"></span>' + label;
        legendDiv.appendChild(item);
    }

    var currentStyle = document.getElementById('style-select').value;
    var isGlassMode = (currentStyle === 'glass' || currentStyle === 'network');

    if (!hasMetadata || !isGlassMode) {
        // Plain Jmol legend: one swatch per element
        var lines = structureData.trim().split('\n');
        var n = parseInt(lines[0], 10);
        var elems = new Set();
        for (var i = 2; i < 2 + n && i < lines.length; i++) {
            var p = lines[i].trim().split(/\s+/);
            if (p.length > 0) elems.add(p[0]);
        }
        Array.from(elems).sort().forEach(function (elem) {
            addSwatch(Jmol[elem] || 0xaaaaaa, elem);
        });
        return;
    }

    // Glass-science legend: formers, then O classes, then modifiers
    var formerElems = new Set();
    var modifierElems = new Set();
    var oClasses = new Set();

    allAtoms.forEach(function (a, idx) {
        var meta = atomMeta[idx] || {};
        var elem = a.elem || a.element || '';
        if (meta.role === 'former') formerElems.add(elem);
        else if (meta.role === 'modifier') modifierElems.add(elem);
        else if (meta.role === 'oxygen' && meta.o_class) oClasses.add(meta.o_class);
    });

    // Formers
    Array.from(formerElems).sort().forEach(function (elem) {
        addSwatch(Jmol[elem] || 0x909090, elem);
    });

    // Oxygen classes (only in network view)
    if (currentStyle === 'network') {
        var oClassLabels = { BO: 'O', NBO: 'O (NBO)', free: 'O (free)', tri: 'O (tricluster)' };
        ['BO', 'NBO', 'free', 'tri'].forEach(function (cls) {
            if (oClasses.has(cls)) {
                addSwatch(O_CLASS_COLORS[cls] || O_DEFAULT_COLOR, oClassLabels[cls] || cls);
            }
        });
    } else {
        // Single O swatch for oxide glass view
        if (oClasses.size > 0) {
            addSwatch(O_DEFAULT_COLOR, 'O');
        }
    }

    // Modifiers (hidden in network-only mode)
    if (currentStyle !== 'network') {
        Array.from(modifierElems).sort().forEach(function (elem) {
            addSwatch(Jmol[elem] || 0x909090, elem);
        });
    }
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
    viewer.removeAllShapes();  // clear custom bonds
    viewer.addModel(structureData, 'xyz');

    // Apply style
    if (style === 'glass') {
        applyGlassStyles(false);
    } else if (style === 'network') {
        applyGlassStyles(true);
    } else {
        viewer.setStyle({}, getStyleForMode(style));
    }

    // If in isolation mode, re-apply isolation
    if (isolatedAtom) {
        isolateAtom(isolatedAtom);
    }

    // Re-add hover functionality
    viewer.setHoverable({}, true, function (atom, viewer, event, container) {
        if (!atom.label) {
            const element = atom.elem || atom.element || 'Unknown';
            const atomIndex = atom.serial || (atom.index !== undefined ? atom.index + 1 : 'N/A');
            const meta = atomMeta[atom.index] || {};
            let tip = element + ' (' + atomIndex + ')';
            if (meta.o_class) tip += ' [' + meta.o_class + ']';
            else if (meta.role) tip += ' [' + meta.role + ']';

            atom.label = viewer.addLabel(tip, {
                position: atom,
                backgroundColor: 'black',
                backgroundOpacity: 0.8,
                fontColor: 'white',
                fontSize: 12,
                borderThickness: 1,
                borderColor: 'black'
            });
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
    buildLegend();
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

    const style = document.getElementById('style-select').value;
    var isGlass = (style === 'glass' || style === 'network');
    var isNetwork = (style === 'network');
    if (isGlass) {
        // Hide all, then show only nearby atoms with glass colors + sticks
        // Bonds are already set in the model by assignGlassBonds()
        viewer.setStyle({}, { sphere: { hidden: true }, stick: { hidden: true } });
        var Jmol = ($3Dmol.elementColors && $3Dmol.elementColors.Jmol) || {};

        // Group near atoms for batched styling
        var groups = {};
        nearIndices.forEach(function (idx) {
            var meta = atomMeta[idx] || {};
            var a = allAtoms[idx];
            var elem = a.elem || a.element || '';
            var color, rad, key, hasStick, hidden;
            if (meta.role === 'oxygen') {
                var cls = meta.o_class || 'default';
                key = 'o_' + cls;
                color = O_CLASS_COLORS[cls] || O_DEFAULT_COLOR;
                rad = RADII.oxygen;
                hasStick = true;
                hidden = false;
            } else if (meta.role === 'former') {
                key = 'f_' + elem;
                color = Jmol[elem] || 0x909090;
                rad = RADII.former;
                hasStick = true;
                hidden = false;
            } else if (meta.role === 'modifier') {
                key = 'm_' + elem;
                color = Jmol[elem] || 0x909090;
                rad = RADII.modifier;
                hasStick = false;
                hidden = isNetwork;
            } else {
                key = 'x_' + elem;
                color = Jmol[elem] || 0x909090;
                rad = RADII.other;
                hasStick = false;
                hidden = isNetwork;
            }
            if (!groups[key]) groups[key] = { indices: [], color: color, radius: rad, hasStick: hasStick, hidden: hidden };
            groups[key].indices.push(idx);
        });
        Object.keys(groups).forEach(function (key) {
            var g = groups[key];
            if (g.hidden) return; // already hidden from the blanket hide-all
            var s = { sphere: { radius: g.radius, color: g.color } };
            if (g.hasStick) s.stick = { radius: STICK_RADIUS, color: 0x888888 };
            viewer.setStyle({ index: g.indices }, s);
        });
    } else {
        // Non-glass modes: hide all, show near with current style
        viewer.setStyle({}, { sphere: { hidden: true }, stick: { hidden: true } });
        var visStyle = getStyleForMode(style);
        nearIndices.forEach(function (idx) {
            viewer.setStyle({ index: idx }, visStyle);
        });
    }

    isolatedAtom = atom;
    document.getElementById('show-all-btn').style.display = 'inline-block';
    viewer.render();
}

// Show all atoms again
function showAllAtoms() {
    if (!viewer) return;
    var style = document.getElementById('style-select').value;
    if (style === 'glass') {
        applyGlassStyles(false);
    } else if (style === 'network') {
        applyGlassStyles(true);
    } else {
        viewer.setStyle({}, getStyleForMode(style));
    }

    isolatedAtom = null;
    document.getElementById('show-all-btn').style.display = 'none';
    addUnitCell();
    viewer.render();
}

// Return the 3Dmol style object for a given mode name
function getStyleForMode(mode) {
    switch (mode) {
        case 'sphere':
            return { sphere: { radius: 0.5, colorscheme: 'Jmol' }, stick: { radius: 0.2, colorscheme: 'Jmol' } };
        case 'stick':
            return { stick: { radius: 0.3, colorscheme: 'Jmol' } };
        case 'vdw':
            return { sphere: { colorscheme: 'Jmol' } };
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
