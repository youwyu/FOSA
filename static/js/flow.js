// Construction figure: replay the most probable edit path each time it scrolls into view.
(() => {
  const flow = document.querySelector('.flow');
  if (!flow || !('IntersectionObserver' in window)) return;
  let visible = false;
  new IntersectionObserver(([entry]) => {
    const now = entry.isIntersecting && entry.intersectionRatio >= .4;
    if (now && !visible) {
      flow.classList.remove('is-running');
      void flow.offsetWidth;
      flow.classList.add('is-running');
    }
    visible = now;
  }, { threshold: [0, .4] }).observe(flow);
})();
