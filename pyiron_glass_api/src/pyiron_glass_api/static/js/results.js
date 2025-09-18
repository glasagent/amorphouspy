// Global variables for 3D visualization
let viewer;
let isSpinning = false;

// Initialize when page loads
window.onload = function() {
    initPlotlyPlot();
    init3DViewer();
};

// Render the Plotly plot
function initPlotlyPlot() {
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
        defaultcolors: $3Dmol.rasmolElementColors
    });

    viewer.addModel(structureData, 'xyz');
    viewer.setStyle({}, {
        sphere: {radius: 0.5},
        stick: {radius: 0.2}
    });

    viewer.setBackgroundColor('white');
    viewer.zoomTo();
    viewer.render();
    console.log('3D viewer initialized successfully');
}

// Update visualization style
function updateStyle() {
    if (!viewer) return;

    const style = document.getElementById('style-select').value;
    viewer.removeAllModels();
    viewer.addModel(structureData, 'xyz');

    switch(style) {
        case 'sphere':
            viewer.setStyle({}, {
                sphere: {radius: 0.5},
                stick: {radius: 0.2}
            });
            break;
        case 'stick':
            viewer.setStyle({}, {
                stick: {radius: 0.3}
            });
            break;
        case 'line':
            viewer.setStyle({}, {
                line: {linewidth: 2}
            });
            break;
        case 'cross':
            viewer.setStyle({}, {
                cross: {radius: 0.5}
            });
            break;
    }
    viewer.render();
}

// Reset camera view
function resetView() {
    if (viewer) {
        viewer.zoomTo();
        viewer.render();
    }
}

// Toggle spinning animation
function toggleSpin() {
    if (!viewer) return;

    if (isSpinning) {
        viewer.spin(false);
        isSpinning = false;
    } else {
        viewer.spin('y', 1);
        isSpinning = true;
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
