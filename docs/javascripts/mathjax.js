window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"], ["$", "$"]],
        displayMath: [["\\[", "\\]"], ["$$", "$$"]],
        processEscapes: true,
        processEnvironments: true
    },
    startup: {
        ready: () => {
            MathJax.startup.defaultReady();
        }
    }
};

document$.subscribe(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
})
