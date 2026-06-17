window.MathJax = {
  loader: {
    load: ["[tex]/boldsymbol"]
  },
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
    packages: {"[+]": ["boldsymbol"]}
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset on every instant-navigation page swap (navigation.instant).
// Clearing MathJax's caches and counters first avoids the stale state that
// otherwise leaves equations unrendered until a full page reload.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
